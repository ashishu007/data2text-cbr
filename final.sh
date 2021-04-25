# echo " "
# echo "Create abstract sentences"
# echo " "
# python3 clustering/create_abs.py num_sample_id_1

# echo " "
# echo "Assign clusters"
# echo " "
# python3 clustering/assign_cluster.py

# echo " "
# echo "Extracting Templates from Sentences"
# echo " "
# python3 src/final_ext_temp.py

# echo " "
# echo "Feature Weighting of Team"
# echo " "
# python3 src/feature_weighting.py team

# echo " "
# echo "Feature Weighting of Player"
# echo " "
# python3 src/feature_weighting.py player

echo " "
echo "Generating Sentences from Templates"
echo " "
python3 src/final_gen.py ca_ftr_wts_gen_new.txt
