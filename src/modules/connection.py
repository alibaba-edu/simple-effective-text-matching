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


@register('none')
def null_connection(x, _, __):
    return x


@register('residual')
def residual(x, res, _):
    if x.shape[-1] != res.shape[-1]:
        x = dense(x, res.shape.as_list()[-1], name='residual_projection')
    return (x + res) * math.sqrt(0.5)


@register('aug')
def augmented_residual(x, res, i):
    if i == 1:
        x = tf.concat([res, x], axis=-1)  # res is embedding
    elif i > 1:
        hidden_size = int(x.shape[-1])
        x = (res[:, :, -hidden_size:] + x) * math.sqrt(0.5)
        x = tf.concat([res[:, :, :-hidden_size], x], axis=-1)  # former half of res is embedding
    return x
