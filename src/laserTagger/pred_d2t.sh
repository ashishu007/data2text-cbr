OUTPUT_DIR=./output
# Download the pretrained BERT model:
# https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
BERT_BASE_DIR=./bert/cased_L-12_H-768_A-12

GPUS=$1

PREDICTION_FILE=./pred/out_new_1.txt
SAVED_MODEL_DIR=./models

# CUDA_VISIBLE_DEVICES=${GPUS} python3 predict_main.py \
echo "im doing it bitch"
python3 predict_main_copy.py \
  --input_file=cbr_out.txt \
  --input_format=discofuse \
  --output_file=${PREDICTION_FILE} \
  --label_map_file=${OUTPUT_DIR}/label_map.txt \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --saved_model=${SAVED_MODEL_DIR}

