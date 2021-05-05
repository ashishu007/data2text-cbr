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

#!/bin/bash

### Required parameters (modify before calling the script!) ###

# Download the WikiSplit data from:
# https://github.com/google-research-datasets/wiki-split
WIKISPLIT_DIR=./data
# Preprocessed data and models will be stored here.
OUTPUT_DIR=./output
# Download the pretrained BERT model:
# https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
BERT_BASE_DIR=./bert/cased_L-12_H-768_A-12

GPUS=$1

### Optional parameters ###

# If you train multiple models on the same data, change this label.
# EXPERIMENT=wikisplit_experiment
EXPERIMENT=discofuse_experiment
# To quickly test that model training works, set the number of epochs to a
# smaller value (e.g. 0.01).
NUM_EPOCHS=3.0
BATCH_SIZE=64
PHRASE_VOCAB_SIZE=500
MAX_INPUT_EXAMPLES=1000000
SAVE_CHECKPOINT_STEPS=500

###########################


# ## 1. Phrase Vocabulary Optimization

# python phrase_vocabulary_optimization.py \
#   --input_file=${WIKISPLIT_DIR}/train.tsv \
#   --input_format=discofuse \
#   --vocabulary_size=${PHRASE_VOCAB_SIZE} \
#   --max_input_examples=${MAX_INPUT_EXAMPLES} \
#   --output_file=${OUTPUT_DIR}/label_map.txt


# # 2. Converting Target Texts to Tags

# python preprocess_main.py \
#   --input_file=${WIKISPLIT_DIR}/dev.tsv \
#   --input_format=discofuse \
#   --output_tfrecord=${OUTPUT_DIR}/tune.tf_record \
#   --label_map_file=${OUTPUT_DIR}/label_map.txt \
#   --vocab_file=${BERT_BASE_DIR}/vocab.txt \
#   --output_arbitrary_targets_for_infeasible_examples=true

# python preprocess_main.py \
#     --input_file=${WIKISPLIT_DIR}/train.tsv \
#     --input_format=discofuse \
#     --output_tfrecord=${OUTPUT_DIR}/train.tf_record \
#     --label_map_file=${OUTPUT_DIR}/label_map.txt \
#     --vocab_file=${BERT_BASE_DIR}/vocab.txt \
#     --output_arbitrary_targets_for_infeasible_examples=false


# ## 3. Model Training

# NUM_TRAIN_EXAMPLES=$(cat "${OUTPUT_DIR}/train.tf_record.num_examples.txt")
# NUM_EVAL_EXAMPLES=$(cat "${OUTPUT_DIR}/tune.tf_record.num_examples.txt")
# CONFIG_FILE=./configs/lasertagger_config.json

# CUDA_VISIBLE_DEVICES=${GPUS} python3 run_lasertagger.py \
#   --training_file=${OUTPUT_DIR}/train.tf_record \
#   --eval_file=${OUTPUT_DIR}/tune.tf_record \
#   --label_map_file=${OUTPUT_DIR}/label_map.txt \
#   --model_config_file=${CONFIG_FILE} \
#   --output_dir=${OUTPUT_DIR}/models/${EXPERIMENT} \
#   --init_checkpoint=${BERT_BASE_DIR}/bert_model.ckpt \
#   --do_train=true \
#   --train_batch_size=${BATCH_SIZE} \
#   --save_checkpoints_steps=${SAVE_CHECKPOINT_STEPS} \
#   --num_train_epochs=${NUM_EPOCHS} \
#   --num_train_examples=${NUM_TRAIN_EXAMPLES} \
#   --num_eval_examples=${NUM_EVAL_EXAMPLES}

#   # --do_eval=true \


## 4. Prediction

echo " "
echo "Prediction"
echo " "
# # Export the model.
# CUDA_VISIBLE_DEVICES=${GPUS} python3 run_lasertagger.py \
#   --label_map_file=${OUTPUT_DIR}/label_map.txt \
#   --model_config_file=${CONFIG_FILE} \
#   --output_dir=${OUTPUT_DIR}/models/${EXPERIMENT} \
#   --do_export=true \
#   --export_path=${OUTPUT_DIR}/models/${EXPERIMENT}/export

# # Get the most recently exported model directory.
# TIMESTAMP=$(ls "${OUTPUT_DIR}/models/${EXPERIMENT}/export/" | \
#             grep -v "temp-" | sort -r | head -1)
# SAVED_MODEL_DIR=${OUTPUT_DIR}/models/${EXPERIMENT}/export/${TIMESTAMP}
# PREDICTION_FILE=${OUTPUT_DIR}/models/${EXPERIMENT}/pred-1.tsv

PREDICTION_FILE=./pred/pred.tsv
SAVED_MODEL_DIR=./models

# CUDA_VISIBLE_DEVICES=${GPUS} python3 predict_main.py \
echo "im doing it bitch"
python3 predict_main.py \
  --input_file=${WIKISPLIT_DIR}/test-1.tsv \
  --input_format=discofuse \
  --output_file=${PREDICTION_FILE} \
  --label_map_file=${OUTPUT_DIR}/label_map.txt \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --saved_model=${SAVED_MODEL_DIR}


## 5. Evaluation

echo " "
echo "Evaluation"
echo " "
python3 score_main.py --prediction_file=${PREDICTION_FILE}
