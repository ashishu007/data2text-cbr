# dynamic-temp-nlg
NLG with dynamic templates

## Process

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
    * Use REG (Referring Expression Generation) here

## How to Run

1. Create the clusters
```bash
sh create_cluster.sh
```

2. Do generation
```bash
sh final.sh
```

## TODO
1. ~~Extract templates for players/teams stats~~
2. ~~Rank/Select important players (currently done based on efficiency)~~
3. ~~Generate sentences for players stats~~
4. ~~Generate sentences for teams stats~~
5. Learn weights of the features
6. Possibly select players' combinations (with other players and with teams)
7. Experiment on lesser training data
8. ~~Apply lm-scoring on template generation (just after clustering maybe)~~
9. ~~Apply lm-scoring on selecting top sentence after generation~~
10. Select clusters based on num-teams and players in the sentences
11. Generate defeat and next-game based on rules
12. If there's a number in the template that doesn't match the stats json, don't take it