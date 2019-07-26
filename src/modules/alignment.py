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


import math
import tensorflow as tf
from functools import partial
from src.utils.registry import register
from . import dense

registry = {}
register = partial(register, registry=registry)


@register('identity')
class Alignment:
    def __init__(self, args):
        self.args = args

    def _attention(self, a, b, t, _):
        return tf.matmul(a, b, transpose_b=True) * t

    def __call__(self, a, b, mask_a, mask_b, dropout_keep_prob, name='alignment'):
        with tf.variable_scope(name):
            temperature = tf.get_variable('temperature', shape=(), dtype=tf.float32, trainable=True,
                                          initializer=tf.constant_initializer(math.sqrt(1 / self.args.hidden_size)))
            tf.summary.histogram('temperature', temperature)
            attention = self._attention(a, b, temperature, dropout_keep_prob)
            attention_mask = tf.matmul(mask_a, mask_b, transpose_b=True)
            attention = attention_mask * attention + (1 - attention_mask) * tf.float32.min
            attention_a = tf.nn.softmax(attention, dim=1)
            attention_b = tf.nn.softmax(attention, dim=2)
            attention_a = tf.identity(attention_a, name='attention_a')
            attention_b = tf.identity(attention_b, name='attention_b')
            tf.summary.histogram('attention_a', tf.boolean_mask(attention_a, tf.cast(attention_mask, tf.bool)))
            tf.summary.histogram('attention_b', tf.boolean_mask(attention_b, tf.cast(attention_mask, tf.bool)))

            feature_b = tf.matmul(attention_a, a, transpose_a=True)
            feature_a = tf.matmul(attention_b, b)
            return feature_a, feature_b


@register('linear')
class MappedAlignment(Alignment):
    def _attention(self, a, b, t, dropout_keep_prob):
        with tf.variable_scope(f'proj'):
            a = dense(tf.nn.dropout(a, dropout_keep_prob),
                      self.args.hidden_size, activation=tf.nn.relu)
            b = dense(tf.nn.dropout(b, dropout_keep_prob),
                      self.args.hidden_size, activation=tf.nn.relu)
            return super()._attention(a, b, t, dropout_keep_prob)
