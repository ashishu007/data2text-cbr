echo " "
echo "Create abstract sentences"
echo " "
python3 ./clustering/create_abs.py

echo " "
echo "Assign clusters"
echo " "
python3 ./clustering/assign_cluster.py
