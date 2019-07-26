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
from . import conv1d


class Encoder:
    def __init__(self, args):
        self.args = args

    def __call__(self, x, mask, dropout_keep_prob, name='encoder'):
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            for i in range(self.args.enc_layers):
                x = mask * x
                if i > 0:
                    x = tf.nn.dropout(x, dropout_keep_prob)
                x = conv1d(x, self.args.hidden_size, kernel_size=self.args.kernel_size, activation=tf.nn.relu,
                           name=f'cnn_{i}')
            x = tf.nn.dropout(x, dropout_keep_prob)
            return x
