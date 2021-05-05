# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

# Lint as: python3
"""Compute realized predictions for a dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import bert_example
import predict_utils
import tagging_converter
import utils

import tensorflow as tf

from nltk import sent_tokenize

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'input_file', None,
    'Path to the input file containing examples for which to compute '
    'predictions.')
flags.DEFINE_enum(
    'input_format', None, ['wikisplit', 'discofuse'],
    'Format which indicates how to parse the input_file.')
flags.DEFINE_string(
    'output_file', None,
    'Path to the TSV file where the predictions are written to.')
flags.DEFINE_string(
    'label_map_file', None,
    'Path to the label map file. Either a JSON file ending with ".json", that '
    'maps each possible tag to an ID, or a text file that has one tag per '
    'line.')
flags.DEFINE_string('vocab_file', None, 'Path to the BERT vocabulary file.')
flags.DEFINE_integer('max_seq_length', 128, 'Maximum sequence length.')
flags.DEFINE_bool(
    'do_lower_case', False,
    'Whether to lower case the input text. Should be True for uncased '
    'models and False for cased models.')
flags.DEFINE_bool('enable_swap_tag', True, 'Whether to enable the SWAP tag.')
flags.DEFINE_string('saved_model', None, 'Path to an exported TF model.')


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  flags.mark_flag_as_required('input_file')
  flags.mark_flag_as_required('input_format')
  flags.mark_flag_as_required('output_file')
  flags.mark_flag_as_required('label_map_file')
  flags.mark_flag_as_required('vocab_file')
  flags.mark_flag_as_required('saved_model')

  label_map = utils.read_label_map(FLAGS.label_map_file)
  converter = tagging_converter.TaggingConverter(
      tagging_converter.get_phrase_vocabulary_from_label_map(label_map),
      FLAGS.enable_swap_tag)
  builder = bert_example.BertExampleBuilder(label_map, FLAGS.vocab_file,
                                            FLAGS.max_seq_length,
                                            FLAGS.do_lower_case, converter)
  predictor = predict_utils.LaserTaggerPredictor(
      tf.contrib.predictor.from_saved_model(FLAGS.saved_model), builder,
      label_map)
  
  summs = open(FLAGS.input_file, 'r').readlines()
  summs = [i.strip() for i in summs]

  new_summs = []
  for idx1, summ in enumerate(summs):
      if idx1 % 10 == 0:
        print(idx1)
      sents = sent_tokenize(summ)
      tmp = [sents[0]]
      for idx, (inco1, inco2) in enumerate(zip(sents[1:-1:2], sents[2::2])): # leave the first sentence out
        preds = predictor.predict([inco1, inco2])
        tmp.append(preds)
      new_summs.append(' '.join(tmp))
  with open(FLAGS.output_file, 'w') as f:
    f.write('\n'.join(new_summs))

#   new_summs = []
#   for idx1, summ in enumerate(summs):
#       if idx1 % 10 == 0:
#         print(idx1)
#       sents = sent_tokenize(summ)
#       new_sum = ""
#       for idx, sent in enumerate(sents):
#         inco1 = new_sum
#         inco2 = sent
#         preds = predictor.predict([inco1, inco2])
#         new_sum = preds
#       new_summs.append(new_sum)
#   with open(FLAGS.output_file, 'w') as f:
#     f.write('\n'.join(new_summs))

if __name__ == '__main__':
  app.run(main)
