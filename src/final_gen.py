import json
from player_stats_temp import generating_player_text_from_templates
from team_stats_temp import generating_team_text_from_templates
from generate_team import generate_defeat_and_next_game

test_preds = []
js = json.load(open(f'./data/jsons/2018_w_opp.json', 'r'))

for game_idx in range(len(js)):

    # if game_idx < 10:
    def_next = generate_defeat_and_next_game(js, game_idx)
    team_stat = generating_team_text_from_templates(js, game_idx)
    player_stat = generating_player_text_from_templates(js, game_idx)

    # print(def_next, team_stat, player_stat)

    sol = [def_next['defeat'].strip()]
    for k, v in team_stat.items():
        sol.append(v.strip())
    for k, v in player_stat.items():
        sol.append(v.strip())
    
    if 'HOME-next_game' in def_next:
        sol.append(def_next['HOME-next_game'].strip())
    if 'VIS-next_game' in def_next:
        sol.append(def_next['VIS-next_game'].strip())

    final_sol = ' '.join(sol)

    if game_idx % 100 == 0:
        print(game_idx, final_sol)

    test_preds.append(final_sol)

with open('./output/only_2014_training_data.txt', 'w') as f:
    f.write('\n'.join(test_preds))

