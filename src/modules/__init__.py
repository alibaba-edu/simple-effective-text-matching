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


import numpy as np
import tensorflow as tf


def gelu(x):
    return 0.5 * x * (1 + tf.nn.tanh(x * 0.7978845608 * (1 + 0.044715 * x * x)))


def get_weight(shape, gain=np.sqrt(2), weight_norm=True, fan_in=None, name='weight'):
    if fan_in is None:
        fan_in = np.prod(shape[:-1])
    std = gain / np.sqrt(fan_in)  # He init

    w = tf.get_variable(name, shape=shape, initializer=tf.initializers.random_normal(0, std),
                        dtype=tf.float32)
    if weight_norm:
        g = tf.get_variable('{}_g'.format(name), shape=(1,) * (len(shape) - 1) + (shape[-1],),
                            initializer=tf.ones_initializer)
        w_norm = tf.sqrt(tf.reduce_sum(tf.square(w), axis=list(range(len(shape) - 1)), keep_dims=True))
        w = w / tf.maximum(w_norm, 1e-7) * g
    return w


def apply_bias(x, name='bias'):
    b = tf.get_variable(name, shape=[x.get_shape()[-1]], initializer=tf.zeros_initializer)
    b = tf.cast(b, x.dtype)
    b = tf.reshape(b, [1] * len(x.get_shape()[:-1]) + [x.get_shape().as_list()[-1]])
    return x + b


def dense(x, units, activation=None, name='dense'):
    with tf.variable_scope(name):
        fan_in = x.shape[-1].value
        new_shape = tf.concat([tf.shape(x)[:-1], tf.constant([units])], axis=0)
        x = tf.reshape(x, (-1, fan_in))
        gain = np.sqrt(2) if activation is tf.nn.relu else 1
        w = get_weight([fan_in, units], gain=gain)
        out = apply_bias(tf.matmul(x, w))
        out = tf.reshape(out, new_shape)
        if activation:
            if activation is tf.nn.relu:
                activation = gelu
            out = activation(out)
        return out


def conv1d(x, filters, kernel_size, activation=None, name='conv1d'):
    with tf.variable_scope(name):
        gain = np.sqrt(2) if activation is tf.nn.relu else 1
        x = tf.expand_dims(x, 1)
        w = get_weight([kernel_size, x.shape[-1].value, filters], gain=gain)
        w = tf.expand_dims(w, 0)
        out = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')
        out = tf.squeeze(out, [1])
        out = apply_bias(out)
        if activation:
            out = activation(out)
        return out
