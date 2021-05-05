echo " "
echo "Create abstract sentences"
echo " "
python3 clustering/create_abs.py num_sample_id_4

echo " "
echo "Assign clusters"
echo " "
python3 clustering/assign_cluster.py

echo " "
echo "Extracting Templates from Sentences"
echo " "
python3 src/extract_template.py

echo " "
echo "Training Important Player classifier and saving data scaler"
echo " "
python3 src/select_imp_players.py

echo " "
echo "Feature Weighting of Team"
echo " "
python3 src/feature_weighting.py team

echo " "
echo "Feature Weighting of Player"
echo " "
python3 src/feature_weighting.py player

OUT_FILE_NAME=info_gain_ftr_wts_gen_basic_atts_all_seasons.txt
echo " "
echo "Generating Sentences from Templates"
echo " "
python3 src/final_gen.py ${OUT_FILE_NAME}

echo " "
echo "LaserTagger"
echo " "
cd src/laserTagger
BERT_BASE_DIR=./bert/cased_L-12_H-768_A-12
SAVED_MODEL_DIR=./models
INPUT_FILE=../../output/${OUT_FILE_NAME}
PREDICTION_FILE=../../output/laserTagger-${OUT_FILE_NAME}
OUTPUT_DIR=./output
python3 predict_main_copy.py \
  --input_file=${INPUT_FILE} \
  --input_format=discofuse \
  --output_file=${PREDICTION_FILE} \
  --label_map_file=${OUTPUT_DIR}/label_map.txt \
  --vocab_file=${BERT_BASE_DIR}/vocab.txt \
  --saved_model=${SAVED_MODEL_DIR}

