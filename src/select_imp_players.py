"""
Select important players from the textual summaries.
"""

import json
import numpy as np
import pandas as pd

def save_train_data_for_imp_players():
    """
    1. also add is_leader feature to the list
    2. possibly add line-scores as well
    """
    player_names = json.load(open('./data/player_names.json', 'r'))
    all_atts = json.load(open('./data/atts.json', 'r'))

    start_season = 14
    while (start_season < 18):
        print(start_season)

        js = json.load(open(f'./data/jsons/20{start_season}_w_opp.json', 'r'))
        print(len(js))

        imp_players_in_game = []
        x, y = [], []
        for item in js:
            summ = item['summary']

            # get the index of all player mentions in the summary
            for tok in summ:
                if tok in player_names['First Names']:
                    for k, v in item['box_score']['FIRST_NAME'].items():
                        if v == tok and k not in imp_players_in_game:
                            imp_players_in_game.append(k)
                elif tok in player_names['Last Names']:
                    for k, v in item['box_score']['SECOND_NAME'].items():
                        if v == tok and k not in imp_players_in_game:
                            imp_players_in_game.append(k)

            player_stats, imp_or_not = [], []
            max_player_in_game = len(item['box_score']['FIRST_NAME'])
            max_player_ftrs = len(all_atts['box-score sim_ftrs keys'])
            # print(max_player_ftrs)
            for player_idx in range(30):
                useful_stats = []
                if player_idx < max_player_in_game:
                    for k, v in item['box_score'].items():
                        if k in all_atts['box-score sim_ftrs keys']:
                            if k not in ['STARTER', 'IS_HOME', 'DOUBLE_DOUBLE']:
                                val = int(v[str(player_idx)]) if v[str(player_idx)] != 'N/A' else 0
                                useful_stats.append(val)
                            else:
                                val = 1 if v == 'yes' else 0
                                useful_stats.append(val)

                    # here is_leader feature can be added
                    if item['box_score']['IS_HOME'][str(player_idx)] == 'yes':
                        val = 1 if item['box_score']['PLAYER_NAME'][str(player_idx)] == f"{item['home_line']['LEADER_FIRST_NAME']} {item['home_line']['LEADER_SECOND_NAME']}" else 0
                    else:
                        val = 1 if item['box_score']['PLAYER_NAME'][str(player_idx)] == f"{item['vis_line']['LEADER_FIRST_NAME']} {item['vis_line']['LEADER_SECOND_NAME']}" else 0
                    useful_stats.append(val)
                
                else:
                    useful_stats.extend([0.0]*max_player_ftrs)
                player_stats.append(useful_stats)
                imp_flag = 1 if str(player_idx) in imp_players_in_game else 0
                imp_or_not.append(imp_flag)
            
            x.append(player_stats)
            y.append(imp_or_not)

        x_arr = np.array(x)
        y_arr = np.array(y)

        # x_arr = num_examples X num_player X num_ftrs ==> (1226, 30, 28)
        # y_arr = num_examples X num_players ==> (1226, 30)
        print(x_arr.shape, y_arr.shape)

        np.save(open(f'./data/imp_players/20{start_season}_x_arr.npy', 'wb'), x_arr)
        np.save(open(f'./data/imp_players/20{start_season}_y_arr.npy', 'wb'), y_arr)

        start_season += 1


def select_imp_player_on_eff(box_score):
    """
    NBA's efficiency rating: (PTS + REB + AST + STL + BLK − ((FGA − FGM) + (FTA − FTM) + TO))
    """
    home_player_eff, vis_player_eff = {}, {}
    for k, v in box_score['PLAYER_NAME'].items():

        pts = int(box_score['PTS'][k]) if box_score['PTS'][k] != 'N/A' else 0
        reb = int(box_score['REB'][k]) if box_score['REB'][k] != 'N/A' else 0
        ast = int(box_score['AST'][k]) if box_score['AST'][k] != 'N/A' else 0
        stl = int(box_score['STL'][k]) if box_score['STL'][k] != 'N/A' else 0
        blk = int(box_score['BLK'][k]) if box_score['BLK'][k] != 'N/A' else 0
        fga = int(box_score['FGA'][k]) if box_score['FGA'][k] != 'N/A' else 0
        fgm = int(box_score['FGM'][k]) if box_score['FGM'][k] != 'N/A' else 0
        fta = int(box_score['FTA'][k]) if box_score['FTA'][k] != 'N/A' else 0
        ftm = int(box_score['FTM'][k]) if box_score['FTM'][k] != 'N/A' else 0
        to = int(box_score['TO'][k]) if box_score['TO'][k] != 'N/A' else 0

        eff = pts + reb + ast + stl + blk - ((fga - fgm) + (fta - ftm) + to)
        if box_score['IS_HOME'][k] == 'yes':
            home_player_eff[v] = eff
        else:
            vis_player_eff[v] = eff

    hpe = dict(sorted(home_player_eff.items(), key=lambda item: item[1], reverse=True))
    vpe = dict(sorted(vis_player_eff.items(), key=lambda item: item[1], reverse=True))        

    return hpe, vpe


# js = json.load(open(f'./data/jsons/2014_w_opp.json', 'r'))
# select_imp_player_on_eff(js[0]['box_score'], js[0]['home_line'], js[0]['vis_line'])

# save_train_data_for_imp_players()