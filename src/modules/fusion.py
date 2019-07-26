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
class Fusion:
    def __init__(self, args):
        super().__init__()
        self.args = args

    def __call__(self, x, align, _):
        with tf.variable_scope('align', reuse=tf.AUTO_REUSE):
            return dense(tf.concat([x, align], axis=-1), self.args.hidden_size,
                         activation=tf.nn.relu)


@register('full')
class FullFusion:
    def __init__(self, args):
        super().__init__()
        self.args = args

    def __call__(self, x, align, dropout_keep_prob):
        with tf.variable_scope('align', reuse=tf.AUTO_REUSE):
            x = tf.concat([
                dense(tf.concat([x, align], axis=-1), self.args.hidden_size, activation=tf.nn.relu, name='orig'),
                dense(tf.concat([x, x - align], axis=-1), self.args.hidden_size, activation=tf.nn.relu, name='sub'),
                dense(tf.concat([x, x * align], axis=-1), self.args.hidden_size, activation=tf.nn.relu, name='mul'),
            ], axis=-1)
            x = tf.nn.dropout(x, dropout_keep_prob)
            x = dense(x, self.args.hidden_size, activation=tf.nn.relu, name="proj")
            return x
