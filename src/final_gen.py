import sys
import json

from player_stats_temp import generating_player_text_from_templates
from team_stats_temp import generating_team_text_from_templates
from generate_team import generate_defeat_and_next_game
from rule_gens import RulesForGeneration

from transformers import GPT2Tokenizer

print("Constructing main file ....")
test_preds = []
js = json.load(open(f'./data/jsons/2018_w_opp.json', 'r'))
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
rfg  = RulesForGeneration()
print("Constructed!!\n\n")

for game_idx in range(len(js)):

    # if game_idx == 0:
    # def_next = generate_defeat_and_next_game(js, game_idx, tokenizer)
    team_stat = generating_team_text_from_templates(js, game_idx, tokenizer)
    player_stat = generating_player_text_from_templates(js, game_idx, tokenizer)

    # print(def_next, team_stat, player_stat)

    # sol = []
    # sol = [def_next['defeat'].strip()]

    sol = [rfg.generate_defeat_sentence(js[game_idx]).strip()]
    
    for k, v in team_stat.items():
        sol.append(v.strip())
    for k, v in player_stat.items():
        sol.append(v.strip())

    # if 'HOME-next_game' in def_next:
    #     sol.append(def_next['HOME-next_game'].strip())
    # if 'VIS-next_game' in def_next:
    #     sol.append(def_next['VIS-next_game'].strip())

    sol.append(rfg.generate_next_game_sentence(js[game_idx]).strip())

    final_sol = ' '.join(sol)

    if game_idx % 10 == 0:
        print()
        print(game_idx, final_sol)

    test_preds.append(final_sol)

with open(f'./output/{sys.argv[1]}', 'w') as f:
    f.write('\n'.join(test_preds))

