import json
import pandas as pd

"""
This file will contain several small chunks of codes to do some non-trivial tasks
"""

class MiscTasks:
    def __init__(self):
        # self.cluster_path = './data/clusters/all_clusters.csv'
        self.cluster_path = './clustering/data/all_clusters.csv'
        self.team_clusts = ['Y', 'F']
        self.player_clusts = ['A', 'D', 'E', 'G', 'H', 'I', 'N', 'O', 'R', 'T', 'V']
        self.defeat_clusts = ['B', 'C']
        self.next_game_clusts = ['J']

    # ------------------------------ Next Opponent -------------------------------------
    def add_next_opponent(self):
        # Here I'm trying to add a feature to define next game's opponent explicitly
        start_season = 14

        while (start_season < 19):
            print(f'{start_season}')
            js = json.load(open(f'./data/jsons/20{start_season}_new_atts_w_stand_streak.json', 'r'))
            # print(js[0].keys(), js[0]['home_next_game'].keys())
            
            new_js = []
            for item in js:
                home_name, vis_name = item['home_name'], item['vis_name']
                if 'home_next_game' in item:
                    home_ng = item['home_next_game'] 
                    if home_name == home_ng['NEXT-HOME-TEAM']:
                        home_next_opponent = home_ng['NEXT-VISITING-TEAM']
                        home_next_opponent_place = home_ng['NEXT-VISITING-TEAM-PLACE']
                        home_next_opponent_conf = home_ng['NEXT-VISITING-TEAM-CONFERENCE']
                        home_next_opponent_div = home_ng['NEXT-VISITING-TEAM-DIVISION']
                    else:
                        home_next_opponent = home_ng['NEXT-HOME-TEAM']
                        home_next_opponent_place = home_ng['NEXT-HOME-TEAM-PLACE']
                        home_next_opponent_conf = home_ng['NEXT-HOME-TEAM-CONFERENCE']
                        home_next_opponent_div = home_ng['NEXT-HOME-TEAM-DIVISION']
                    item['home_next_game']['NEXT-OPPONENT-TEAM'] = home_next_opponent
                    item['home_next_game']['NEXT-OPPONENT-TEAM-PLACE'] = home_next_opponent_place
                    item['home_next_game']['NEXT-OPPONENT-TEAM-CONFERENCE'] = home_next_opponent_conf
                    item['home_next_game']['NEXT-OPPONENT-TEAM-DIVISION'] = home_next_opponent_div

                if 'vis_next_game' in item:
                    vis_ng = item['vis_next_game']
                    if vis_name == vis_ng['NEXT-HOME-TEAM']:
                        vis_next_opponent = vis_ng['NEXT-VISITING-TEAM']
                        vis_next_opponent_place = vis_ng['NEXT-VISITING-TEAM-PLACE']
                        vis_next_opponent_conf = vis_ng['NEXT-VISITING-TEAM-CONFERENCE']
                        vis_next_opponent_div = vis_ng['NEXT-VISITING-TEAM-DIVISION']
                    else:
                        vis_next_opponent = vis_ng['NEXT-HOME-TEAM']
                        vis_next_opponent_place = vis_ng['NEXT-HOME-TEAM-PLACE']
                        vis_next_opponent_conf = vis_ng['NEXT-HOME-TEAM-CONFERENCE']
                        vis_next_opponent_div = vis_ng['NEXT-HOME-TEAM-DIVISION']
                    item['vis_next_game']['NEXT-OPPONENT-TEAM'] = vis_next_opponent
                    item['vis_next_game']['NEXT-OPPONENT-TEAM-PLACE'] = vis_next_opponent_place
                    item['vis_next_game']['NEXT-OPPONENT-TEAM-CONFERENCE'] = vis_next_opponent_conf
                    item['vis_next_game']['NEXT-OPPONENT-TEAM-DIVISION'] = vis_next_opponent_div
                new_js.append(item)
            
            # print(new_js[0]['home_next_game'], new_js[0]['home_name'])
            json.dump(new_js, open(f'./data/jsons/20{start_season}_w_opp.json', 'w'))

            start_season += 1
    # ------------------------------ Next Opponent -------------------------------------


    # ------------------------------ Replace Ordinal Numbers -------------------------------------
    def replace_ordinals_from_text(self):
        # here i'm trying to replace all the occurence of ordinals from texts into their cardinal counterparts

        ordinal_set = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9,
                            "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
                            "1st": 1, "2nd": 2, "3rd": 3, "4th": 4, "5th": 5, "6th": 6, "7th": 7, "8th": 8, "9th": 9, "10th": 10, 
                            "11th": 11, "12th": 12, "13th": 13, "14th": 14, "15th": 15, "last": 15, "atop":1}

        df = pd.read_csv('./data/clusters/sents_from_code_w_clusts.csv')

        new_sents_wo_ord = []
        for _, row in df.iterrows():
            sent = row['sent']
            new_sent = ''
            for tok in sent.split(' '):
                if tok in ordinal_set:
                    new_sent += f'{ordinal_set[tok]} '
                else:
                    new_sent += f'{tok} '
            new_sents_wo_ord.append(new_sent)

        df['sent_wo_ord'] = new_sents_wo_ord

        df.to_csv('./data/clusters/all_clusters.csv', index=0)

    # ------------------------------ Replace Ordinal Numbers -------------------------------------


    # ------------------------------ Unique Player Names -------------------------------------
    def save_unique_player_names(self):
        start_season = 14

        full_names = []
        first_names = []
        last_names = []

        while (start_season < 19):
            js = json.load(open(f'./data/jsons/20{start_season}_w_opp.json', 'r'))
            for item in js:
                bs = item['box_score']
                full_names.extend(list(bs['PLAYER_NAME'].values()))
                first_names.extend(list(bs['FIRST_NAME'].values()))
                last_names.extend(list(bs['SECOND_NAME'].values()))
            start_season += 1

        unique_full_names = list(set(full_names))
        unique_last_names = list(set(last_names))
        unique_first_names = list(set(first_names))

        names = {
            "Full Names": unique_full_names,
            "Last Names": unique_last_names,
            "First Names": unique_first_names
        }

        json.dump(names, open('./data/player_names.json', 'w'), indent='\t')
    # ------------------------------ Unique Player Names -------------------------------------


    # ------------------------------ Team Names -------------------------------------
    def get_all_team_names(self):
        start_season = 14

        team_names = []
        team_places = []
        full_team_names = []

        while (start_season < 19):
            js = json.load(open(f'./data/jsons/20{start_season}_w_opp.json', 'r'))
            for item in js:
                team_names.append(item['vis_name'])
                team_places.append(item['vis_city'])
                full_team_names.append(f'{item["vis_city"]} {item["vis_name"]}')
                team_names.append(item['home_name'])
                team_places.append(item['home_city'])
                full_team_names.append(f'{item["home_city"]} {item["home_name"]}')
            start_season += 1

        names = {
            "Team Names": list(set(team_names)),
            "Team Places": list(set(team_places)),
            "Full Team Names": list(set(full_team_names))
        }

        json.dump(names, open('./data/team_names.json', 'w'), indent='\t')
    # ------------------------------ Team Names -------------------------------------
