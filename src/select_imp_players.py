"""
Select important players from the textual summaries.
"""

import json
import numpy as np

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
        max_player_ftrs = len(all_atts['box-score keys'])
        for player_idx in range(30):
            useful_stats = []
            if player_idx < max_player_in_game:
                for k, v in item['box_score'].items():
                    if k in all_atts['box-score keys']:
                        if k not in ['STARTER', 'IS_HOME', 'DOUBLE_DOUBLE']:
                            val = float(v[str(player_idx)]) if v[str(player_idx)] != 'N/A' else 0.0
                            useful_stats.append(val)
                        else:
                            val = 1.0 if v == 'yes' else 0.0
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
