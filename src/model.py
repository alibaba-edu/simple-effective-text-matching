# coding=utf-8
# Copyright (C) 2019 Alibaba Group Holding Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import re
import sys
import random
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.python.ops.lookup_ops import HashTable
from tensorflow.python.ops.lookup_ops import TextFileIdTableInitializer
from tensorflow.python.client import device_lib

from .network import Network
from .utils.vocab import Vocab
from .utils.metrics import registry as metrics


class Model:
    prefix = 'checkpoint'
    best_model_name = 'best'

    def __init__(self, args, session, updates=None):
        self.args = args
        self.sess = session

        # updates
        if not updates:
            updates = 0
        self.updates = updates
        self.global_step = tf.get_variable('global_step', shape=(), dtype=tf.float32,
                                           initializer=tf.constant_initializer(updates), trainable=False)
        self.step = tf.assign_add(self.global_step, 1)

        # placeholders
        table = HashTable(TextFileIdTableInitializer(filename=os.path.join(args.output_dir, 'vocab.txt')),
                          default_value=Vocab.unk())
        self.q1_string = tf.placeholder(tf.string, [None, None], name='q1_str')
        self.q2_string = tf.placeholder(tf.string, [None, None], name='q2_str')
        self.q1 = tf.placeholder_with_default(table.lookup(self.q1_string), [None, None], name='q1')
        self.q2 = tf.placeholder_with_default(table.lookup(self.q2_string), [None, None], name='q2')
        self.q1_len = tf.placeholder(tf.int32, [None], name='q1_len')
        self.q2_len = tf.placeholder(tf.int32, [None], name='q2_len')
        self.y = tf.placeholder(tf.int32, [None], name='y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, (), name='dropout_keep_prob')

        q1_mask = tf.expand_dims(tf.sequence_mask(self.q1_len, dtype=tf.float32), dim=-1)
        q2_mask = tf.expand_dims(tf.sequence_mask(self.q2_len, dtype=tf.float32), dim=-1)

        devices = self.get_available_gpus() or ['/device:CPU:0']
        if not args.multi_gpu:
            devices = devices[:1]
        if len(devices) == 1:
            splits = 1
        else:
            splits = [tf.shape(self.q1)[0] // len(devices)] * (len(devices) - 1) + [-1]  # handle uneven split
        q1 = tf.split(self.q1, splits)
        q2 = tf.split(self.q2, splits)
        q1_mask = tf.split(q1_mask, splits)
        q2_mask = tf.split(q2_mask, splits)
        y = tf.split(self.y, splits)

        # network
        self.network = Network(args)

        # optimizer
        lr = tf.get_variable('lr', shape=(), dtype=tf.float32,
                             initializer=tf.constant_initializer(args.lr), trainable=False)
        lr_next = tf.cond(self.global_step < args.lr_warmup_steps,
                          true_fn=lambda: args.min_lr +
                                          (args.lr - args.min_lr) / max(1, args.lr_warmup_steps) * self.global_step,
                          false_fn=lambda: tf.maximum(args.min_lr, args.lr * args.lr_decay_rate ** tf.floor(
                              (self.global_step - args.lr_warmup_steps) / args.lr_decay_steps)))
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, tf.assign(lr, lr_next, name='update_lr'))
        self.lr = lr
        self.opt = tf.train.AdamOptimizer(learning_rate=lr, beta1=args.beta1, beta2=args.beta2)

        # training graph
        tower_names = ['tower-{}'.format(i) for i in range(len(devices))] if len(devices) > 1 else ['']
        tower_logits = []
        tower_grads = []
        summaries = []
        loss = 0

        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            for i, device in enumerate(devices):
                with tf.device(device):
                    with tf.name_scope(tower_names[i]) as scope:
                        logits = self.network(q1[i], q2[i], q1_mask[i], q2_mask[i], self.dropout_keep_prob)
                        tower_logits.append(logits)
                        loss = self.get_loss(logits, y[i])
                        tf.get_variable_scope().reuse_variables()
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        grads = self.opt.compute_gradients(loss)
                        tower_grads.append(grads)
        gradients = []
        variables = []
        for grad_and_vars in zip(*tower_grads):
            if grad_and_vars[0][0] is None:
                msg = 'WARNING: trainable variable {} receives no grad.\n'.format(grad_and_vars[0][1].op.name)
                sys.stderr.write(msg)
                continue
            grad = tf.stack([g for g, _ in grad_and_vars])
            grad = tf.reduce_mean(grad, 0)
            v = grad_and_vars[0][1]  # use the first tower's pointer to the (shared) variable
            gradients.append(grad)
            variables.append(v)

        gradients, self.gnorm = tf.clip_by_global_norm(gradients, self.args.grad_clipping)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = self.opt.apply_gradients(zip(gradients, variables))
        logits = tf.concat(tower_logits, 0)
        self.prob = tf.nn.softmax(logits, dim=1, name='prob')
        self.pred = tf.argmax(input=logits, axis=1, name='pred')
        self.loss = tf.identity(loss, name='loss')
        summaries.append(tf.summary.scalar('training/lr', lr))
        summaries.append(tf.summary.scalar('training/gnorm', self.gnorm))
        summaries.append(tf.summary.scalar('training/loss', self.loss))

        # add summary
        self.summary = tf.summary.merge(summaries)

        # saver
        self.saver = tf.train.Saver([var for var in tf.global_variables() if 'Adam' not in var.name],
                                    max_to_keep=args.max_checkpoints)

    def update(self, sess, batch):
        feed_dict = self.process_data(batch, training=True)
        _, gnorm, loss, summary, lr = sess.run(
            [self.train_op, self.gnorm, self.loss, self.summary, self.lr],
            feed_dict=feed_dict
        )
        assert gnorm >= 0, 'encounter nan in gradients.'
        sess.run(self.step)
        self.updates += 1
        stats = {
            'updates': self.updates,
            'loss': loss,
            'lr': lr,
            'gnorm': gnorm,
            'summary': summary,
        }
        return stats

    def evaluate(self, sess, data):
        predictions = []
        targets = []
        probabilities = []
        losses = []
        for batch in data:
            feed_dict = self.process_data(batch, training=False)
            loss, pred, prob = sess.run(
                [self.loss, self.pred, self.prob],
                feed_dict=feed_dict
            )
            losses.append(loss)
            predictions.extend(pred.tolist())
            targets.extend(feed_dict[self.y])
            probabilities.extend(prob.tolist())
        outputs = {
            'target': targets,
            'prob': probabilities,
            'pred': predictions,
            'args': self.args,
        }
        stats = {
            'updates': self.updates,
            'loss': sum(losses[:-1]) / (len(losses) - 1) if len(losses) > 1 else sum(losses),
        }
        for metric in self.args.watch_metrics:
            if metric not in stats:  # multiple metrics could be computed by the same function
                stats.update(metrics[metric](outputs))
        assert 'score' not in stats, 'metric name collides with "score"'
        eval_score = stats[self.args.metric]
        stats['score'] = eval_score
        return eval_score, stats  # first value is for early stopping

    def predict(self, sess, batch):
        feed_dict = self.process_data(batch, training=False)
        return sess.run(self.prob, feed_dict=feed_dict)

    def process_data(self, batch, training):
        feed_dict = {
            self.q1: batch['text1'],
            self.q2: batch['text2'],
            self.q1_len: batch['len1'],
            self.q2_len: batch['len2'],
            self.dropout_keep_prob: self.args.dropout_keep_prob,
        }
        if not training:
            feed_dict[self.dropout_keep_prob] = 1.
        if 'target' in batch:
            feed_dict[self.y] = batch['target']
        return feed_dict

    @staticmethod
    def get_loss(logits, target):
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=target)
        return tf.reduce_mean(losses)

    def save(self, states, name=None):
        self.saver.save(self.sess, os.path.join(self.args.summary_dir, self.prefix),
                        global_step=self.updates)
        if not name:
            name = str(self.updates)
        # noinspection PyTypeChecker
        numpy_state = list(np.random.get_state())
        numpy_state[1] = numpy_state[1].tolist()
        params = {
            'updates': self.updates,
            'args': self.args,
            'random_state': random.getstate(),
            'numpy_state': numpy_state,
        }
        params.update(states)
        with open(os.path.join(self.args.summary_dir, '{}-{}.stat'.format(self.prefix, name)), 'wb') as f:
            pickle.dump(params, f)

    @classmethod
    def load(cls, sess, model_path):
        with open(model_path + '.stat', 'rb') as f:
            checkpoint = pickle.load(f)
        args = checkpoint['args']
        args.summary_dir = os.path.dirname(model_path)
        args.output_dir = os.path.dirname(args.summary_dir)
        model = cls(args, sess, updates=checkpoint['updates'])

        init_vars = tf.train.list_variables(model_path)
        model_vars = {re.match("^(.*):\\d+$", var.name).group(1): var for var in tf.global_variables()}
        assignment_map = {name: model_vars[name] for name, _ in init_vars if name in model_vars}
        tf.train.init_from_checkpoint(model_path, assignment_map)
        sess.run(tf.global_variables_initializer())
        return model, checkpoint

    @staticmethod
    def num_parameters(exclude_embed=False):
        num_params = int(np.sum([np.prod(v.shape.as_list()) for v in tf.trainable_variables()]))
        if exclude_embed:
            emb_params = int(np.sum([np.prod(v.shape.as_list())
                                     for v in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                scope='embedding')]))
            num_params -= emb_params
        return num_params

    def set_embeddings(self, sess, embeddings):
        self.network.embedding.set_(sess, embeddings)

    @staticmethod
    def get_available_gpus():
        local_device_protos = device_lib.list_local_devices()
        return [x.name for x in local_device_protos if x.device_type == 'GPU']
