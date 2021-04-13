echo " "
echo "Extracting Team-Templates from Sentences"
echo " "
python3 ./src/ext_team_temp.py

echo " "
echo "Generating Team-Sentences from Templates"
echo " "
python3 ./src/generate_team.py
