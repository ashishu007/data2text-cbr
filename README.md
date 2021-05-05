# Data2Text-CBR
Data-to-Text Generation with Case-Based Reasoning

## Requirements

Run:
```bash
pip install -r requirements.txt
```

<!-- ## Process

### Create Clusters
1. Cluster the messages/sentences from training summaries

### Template Extraction
1. The whole summary can be divided into following components:
    * Winner: Team A defeated Team B
    * Team stats: some stats about both the teams
    * Player stats: some stats about some players (choose important ones)
        * To select important players, learn a model
        * model will predict: for each player, if it should be added in the summary or not - based on its performance (numercial features)
    * Next Game: teams' next games
2. The clusters, also convey these patterns within themselves
3. Extract templates for these components separately from their corresponding clusters 
4. Save the templates on solution side while features (numerical stats) of the corresponding team/player on the problem side

## Generation
1. Break the target problem into components as well
2. Use the same features saved in problem side in case-base to calculate similarity between target problem
    * Learn the weights of the features
3. Use the top most similar template to generate the solution for each component
    * Use LM-Scoring to select the best out of 5/10
4. Combine the solutions
    * Use REG (Referring Expression Generation) here -->

## How to Run

<!-- 1. Create the clusters
```bash
sh create_cluster.sh
``` -->

<!-- 2. Do generation
```bash
sh final.sh
``` -->

### Download the GPT2 finetuned model

Download the fine-tuned GPT2 model from [GDrive](https://drive.google.com/drive/folders/11q4pXX_MPB8P-XNdDfznq9KhnhcMZqol?usp=sharing).
It's a zip folder, unzip the files into a `gpt2-finetuned` folder in root directory.

### Download the LaserTagger model

Download the trained LaserTagger model from [GDrive](https://drive.google.com/file/d/1uZI-ozhOj2KwzDjZDbgTro2JplDLGSXA/view?usp=sharing). Put the contens of this zip folder into `src/laserTagger/models` folder.

You'll also need to download a pretrained BERT model from the [official repository](https://github.com/google-research/bert#pre-trained-models).
You need to download the 12-layer ''BERT-Base, Cased'' model. Put the contents inside `src/laserTagger/bert` folder.

Note: there might be some issues with the TensorFlow version used in LaserTagger. You might need to run it in a virtua-environment then. Anyhow, even without LaserTagger generation can be done and there won't be any noticable difference in the metric scores.

### Run

```bash
sh final.sh
```

1. Create clusters
2. Train Feature Weighting
3. Train important player classifier
4. Create Case-Base
5. Do generation
6. Apply LaserTagger

<!-- 
## TODO
1. ~~Extract templates for players/teams stats~~
2. ~~Rank/Select important players (currently done based on efficiency)~~
3. ~~Generate sentences for players stats~~
4. ~~Generate sentences for teams stats~~
5. ~~Learn weights of the features~~ 
6. Possibly select players' combinations (with other players and with teams)
7. ~~Experiment on lesser training data~~
8. ~~Apply lm-scoring on template generation (just after clustering maybe)~~
9. ~~Apply lm-scoring on selecting top sentence after generation~~
10. Select clusters based on num-teams and players in the sentences
11. ~~Generate defeat and next-game based on rules~~
12. If there's a number in the template that doesn't match the stats json, don't take the template
13. Plot cluster paths 
14. ~~Scale the input (sim ftrs) data~~
15. ~~Use alignment for feature importance~~
16. How many summaries follow the defined higher-leve template
17. Improve template generation with codes from rw_fg
18. Experiment on basic attributes data

## LM-Scoring
The [lm-scorer](https://github.com/simonepri/lm-scorer) library doesn't work on Python 3.8+ so I used GPT2 to score sentences.
I fine-tuned a GPT2 model on the Rotowire data using [huggingface's transformers](https://github.com/huggingface/transformers) library and used the model to get perplexity for each proposed solution. -->


<!-- 
# current -> H : next -> H
self.ngt1 = "The TEAM-NAME will stay home to host NEXT-OPPONENT-TEAM on NEXT-DAYNAME ."
# current -> H/V : next -> H : win_streak -> >2
self.ngt2 = "The TEAM-NAME will look to continue their winning streak when they host NEXT-OPPONENT-TEAM on NEXT-DAYNAME ."
# current -> V : next -> V
self.ngt3 = "Next , the TEAM-NAME will head to NEXT-OPPONENT-TEAM-PLACE to face NEXT-OPPONENT-TEAM on NEXT-DAYNAME ."
# current -> H : next -> V
self.ngt4 = "TEAM-PLACE TEAM-NAME will take on the NEXT-OPPONENT-TEAM on NEXT-DAYNAME in NEXT-OPPONENT-TEAM-PLACE next scheduled game ."
# current -> H/V : next -> V
self.ngt5 = "The TEAM-NAME now head to NEXT-OPPONENT-TEAM-PLACE for a NEXT-DAYNAME night showdown versus the NEXT-OPPONENT-TEAM ."
# current -> H/V : next -> V
self.ngt6 = "The TEAM-NAME head to NEXT-OPPONENT-TEAM-PLACE to face off against the NEXT-OPPONENT-TEAM on NEXT-DAYNAME night ."
# current -> H : next -> V
self.ngt7 = "The TEAM-NAME travel to NEXT-OPPONENT-TEAM-PLACE for a NEXT-DAYNAME tilt versus the NEXT-OPPONENT-TEAM ."
# current -> H/V : next -> H : this_game -> lost
self.ngt8 = "The TEAM-NAME will look to bounce back when they host NEXT-OPPONENT-TEAM-PLACE NEXT-OPPONENT-TEAM on NEXT-DAYNAME ."
# current -> H/V : next -> H
self.ngt9 = "TEAM-NAME will host the NEXT-OPPONENT-TEAM-PLACE NEXT-OPPONENT-TEAM on NEXT-DAYNAME ."
# current -> V : next -> H 
self.ngt10 = "The TEAM-PLACE TEAM-NAME will return home to face the NEXT-OPPONENT-TEAM on NEXT-DAYNAME ."
# current -> H/V : next -> H : duration betn this & next game -> >3
self.ngt11 = "The TEAM-PLACE TEAM-NAME now have a couple days off , before they play host to the NEXT-OPPONENT-TEAM on NEXT-DAYNAME . "
# current -> H/V : next -> H/V : this_game -> lost
self.ngt12 = "The TEAM-NAME will look to bounce back when they take on NEXT-OPPONENT-TEAM-PLACE NEXT-OPPONENT-TEAM on NEXT-DAYNAME ."
 -->

<!-- 
# 1. current -> H : next -> H
# 2. current -> H/V : next -> H : win_streak -> >2
# 3. current -> V : next -> V
# 4. current -> H/V : next -> V
# 5. current -> H/V : next -> H : this_game -> lost
# 6. current -> V : next -> H 
# 7. current -> H/V : next -> H : duration betn this & next game -> >3
# 8. current -> H/V : next -> H/V : this_game -> lost
 -->
