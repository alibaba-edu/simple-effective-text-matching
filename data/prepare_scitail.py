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


import re
import os
import json
from tqdm import tqdm
from nltk.tokenize import TweetTokenizer


tokenizer = TweetTokenizer()
label_map = {
    'entailment': 0,
    'neutral': 1,
    'contradiction': 2,
}


def tokenize(string):
    string = ' '.join(tokenizer.tokenize(string))
    string = re.sub(r"[-.#\"/]", " ", string)
    string = re.sub(r"\'(?!(s|m|ve|t|re|d|ll)( |$))", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'m", " \'m", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


os.makedirs('scitail', exist_ok=True)


for split in ['train', 'dev', 'test']:
    print('processing SciTail', split)
    with open('orig/SciTailV1.1/snli_format/scitail_1.0_{}.txt'.format(split)) as f, \
            open('scitail/{}.txt'.format(split), 'w', encoding='utf8') as fout:
        n_lines = 0
        for _ in f:
            n_lines += 1
        f.seek(0)
        for line in tqdm(f, total=n_lines, desc=split, leave=False):
            sample = json.loads(line)
            sentence1 = tokenize(sample['sentence1'])
            sentence2 = tokenize(sample['sentence2'])
            label = sample["gold_label"]
            assert label in label_map
            label = label_map[label]
            fout.write('{}\t{}\t{}\n'.format(sentence1, sentence2, label))
