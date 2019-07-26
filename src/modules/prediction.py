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


import tensorflow as tf
from functools import partial
from src.utils.registry import register
from . import dense

registry = {}
register = partial(register, registry=registry)


@register('simple')
class Prediction:
    def __init__(self, args):
        self.args = args

    def _features(self, a, b):
        return tf.concat([a, b], axis=-1)

    def __call__(self, a, b, dropout_keep_prob, name='prediction'):
        x = self._features(a, b)
        with tf.variable_scope(name):
            x = tf.nn.dropout(x, dropout_keep_prob)
            x = dense(x, self.args.hidden_size, activation=tf.nn.relu, name='dense_1')
            x = tf.nn.dropout(x, dropout_keep_prob)
            x = dense(x, self.args.num_classes, activation=None, name='dense_2')
            return x


@register('full')
class AdvancedPrediction(Prediction):
    def _features(self, a, b):
        return tf.concat([a, b, a * b, a - b], axis=-1)


@register('symmetric')
class SymmetricPrediction(Prediction):
    def _features(self, a, b):
        return tf.concat([a, b, a * b, tf.abs(a - b)], axis=-1)
