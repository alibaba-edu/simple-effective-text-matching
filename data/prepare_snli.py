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
import json
import string
import numpy as np
import msgpack
from collections import Counter

in_dir = 'orig/SNLI'
out_dir = '../models/snli/'
data_dir = 'snli'
label_map = {2: '0', 1: '1', 0: '2'}

os.makedirs(out_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
with open(os.path.join(in_dir, 'env')) as f:
    env = json.load(f)

print('convert embeddings ...')
emb = np.load(os.path.join(in_dir, 'emb_glove_300.npy'))
print(len(emb))
with open(os.path.join(out_dir, 'embedding.msgpack'), 'wb') as f:
    msgpack.dump(emb.tolist(), f)

print('convert_vocab ...')
w2idx = env['word_index']
print(len(w2idx))
idx2w = {i: w for w, i in w2idx.items()}
with open(os.path.join(out_dir, 'vocab.txt'), 'w') as f:
    for index in range(len(idx2w)):
        if index >= 2:
            f.write('{}\n'.format(idx2w[index]))
with open(os.path.join(out_dir, 'target_map.txt'), 'w') as f:
    for label in (0, 1, 2):
        f.write('{}\n'.format(label))

# save data files
punctuactions = set(string.punctuation)
for split in ['train', 'dev', 'test']:
    labels = Counter()
    print('convert', split, '...')
    data = env[split]
    with open(os.path.join(data_dir, f'{split}.txt'), 'w') as f_out:
        for sample in data:
            a, b, label = sample
            a = a[1:-1]
            b = b[1:-1]
            a = [w.lower() for w in a if w and w not in punctuactions]
            b = [w.lower() for w in b if w and w not in punctuactions]
            assert all(w in w2idx for w in a) and all(w in w2idx for w in b)
            a = ' '.join(a)
            b = ' '.join(b)
            assert len(a) != 0 and len(b) != 0
            labels.update({label: 1})
            assert label in label_map
            label = label_map[label]
            f_out.write(f'{a}\t{b}\t{label}\n')
    print('labels:', labels)
