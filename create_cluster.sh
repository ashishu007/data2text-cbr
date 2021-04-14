echo " "
echo "Create abstract sentences"
echo " "
python3 ./clustering/create_abs.py num_sample_id_1

echo " "
echo "Assign clusters"
echo " "
python3 ./clustering/assign_cluster.py
