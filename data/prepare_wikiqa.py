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
from shutil import copyfile


def copy(src, tgt):
    copyfile(os.path.abspath(src), os.path.abspath(tgt))


os.makedirs('wikiqa', exist_ok=True)


copy('orig/WikiQACorpus/WikiQA-dev-filtered.ref', 'wikiqa/dev.ref')
copy('orig/WikiQACorpus/WikiQA-test-filtered.ref', 'wikiqa/test.ref')
copy('orig/WikiQACorpus/emnlp-table/WikiQA.CNN.dev.rank', 'wikiqa/dev.rank')
copy('orig/WikiQACorpus/emnlp-table/WikiQA.CNN.test.rank', 'wikiqa/test.rank')
for split in ['train', 'dev', 'test']:
    print('processing WikiQA', split)
    copy('orig/WikiQACorpus/WikiQA-{}.txt'.format(split), 'wikiqa/{}.txt'.format(split))
