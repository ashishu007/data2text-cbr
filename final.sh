echo " "
echo "Extracting Templates from Sentences"
echo " "
python3 ./src/final_ext_temp.py

echo " "
echo "Generating Sentences from Templates"
echo " "
python3 ./src/final_gen.py only_2014.txt
