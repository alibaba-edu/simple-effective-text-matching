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
from .modules.embedding import Embedding
from .modules.encoder import Encoder
from .modules.alignment import registry as alignment
from .modules.fusion import registry as fusion
from .modules.connection import registry as connection
from .modules.pooling import pooling
from .modules.prediction import registry as prediction


class Network:
    def __init__(self, args):
        self.embedding = Embedding(args)
        self.blocks = [{
            'encoder': Encoder(args),
            'alignment': alignment[args.alignment](args),
            'fusion': fusion[args.fusion](args),
        } for _ in range(args.blocks)]
        self.connection = connection[args.connection]
        self.pooling = pooling
        self.prediction = prediction[args.prediction](args)

    def __call__(self, a, b, mask_a, mask_b, dropout_keep_prob):
        a = self.embedding(a, dropout_keep_prob)
        b = self.embedding(b, dropout_keep_prob)
        res_a, res_b = a, b

        for i, block in enumerate(self.blocks):
            with tf.variable_scope('block-{}'.format(i), reuse=tf.AUTO_REUSE):
                if i > 0:
                    a = self.connection(a, res_a, i)
                    b = self.connection(b, res_b, i)
                    res_a, res_b = a, b
                a_enc = block['encoder'](a, mask_a, dropout_keep_prob)
                b_enc = block['encoder'](b, mask_b, dropout_keep_prob)
                a = tf.concat([a, a_enc], axis=-1)
                b = tf.concat([b, b_enc], axis=-1)
                align_a, align_b = block['alignment'](a, b, mask_a, mask_b, dropout_keep_prob)
                a = block['fusion'](a, align_a, dropout_keep_prob)
                b = block['fusion'](b, align_b, dropout_keep_prob)
        a = self.pooling(a, mask_a)
        b = self.pooling(b, mask_b)
        return self.prediction(a, b, dropout_keep_prob)
