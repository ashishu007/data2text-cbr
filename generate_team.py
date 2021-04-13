import json
import pandas as pd
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances

"""
0. Divide data
1. Rank the templates from the case-base
2. Fill the templates
"""

defeat_sim_ftrs_keys = ["TEAM-PTS_QTR1", "TEAM-PTS_QTR2", "TEAM-PTS_QTR3", "TEAM-PTS_QTR4", "TEAM-PTS", "TEAM-FGM", "TEAM-FGA", "TEAM-FG_PCT",
		        "TEAM-FG3M", "TEAM-FG3A", "TEAM-FG3_PCT", "TEAM-FTM", "TEAM-FTA", "TEAM-FT_PCT", "TEAM-REB", "TEAM-AST", "TEAM-TOV", 
        		"TEAM-WINS", "TEAM-LOSSES", "WINNER", "TEAM-STANDING", "STREAK-TYPE", "STREAK-COUNT"]
home_next_game_sim_ftrs_keys = ["HOME-NEXT-HOME", "HOME-NEXT-VIS"]
vis_next_game_sim_ftrs_keys = ["VIS-NEXT-HOME", "VIS-NEXT-VIS"]

all_atts = json.load(open('atts.json', 'r'))

nick_names = {"Sixers": "76ers", "Cavs": "Cavaliers", "T'wolves": "Timberwolves"}

test_json = json.load(open(f'./data/jsons/2018_new_atts_w_stand_streak.json', 'r'))
game_idx = 11

hl = test_json[game_idx]['home_line']
vl = test_json[game_idx]['vis_line']
g = test_json[game_idx]['game']

next_game_exist = False
if 'home_next_game' in test_json[game_idx]:
    hn = test_json[game_idx]['home_next_game']
    vn = test_json[game_idx]['vis_next_game']
    next_game_exist = True

ls = {}
for k, v in hl.items():
    if k in all_atts['line-score keys']:
        if k in defeat_sim_ftrs_keys:
            if k == 'WINNER':
                ls[f'HOME-{k}'] = 0.0 if v == 'no' else 1.0
                ls[f'VIS-{k}'] = 0.0 if vl[k] == 'no' else 1.0
            elif k == 'STREAK-TYPE':
                ls[f'HOME-{k}'] = 0.0 if v == 'L' else 1.0
                ls[f'VIS-{k}'] = 0.0 if vl[k] == 'L' else 1.0
            else:
                ls[f"HOME-{k}"] = float(v)
                ls[f"VIS-{k}"] = float(vl[k])
        else:
            ls[f'HOME-{k}'] = v
            ls[f'VIS-{k}'] = vl[k]

for k, v in g.items():
    if k in all_atts['game keys']:
        ls[f'{k}'] = v

if next_game_exist:
    for k, v in hn.items():
        if k in all_atts['next-game keys']:
            if hl['TEAM-NAME'] == hn['NEXT-HOME-TEAM']:
                ls['HOME-NEXT-HOME'] = 1.0
                ls['HOME-NEXT-VIS'] = 0.0
            if hl['TEAM-NAME'] == hn['NEXT-VISITING-TEAM']:
                ls['HOME-NEXT-HOME'] = 0.0
                ls['HOME-NEXT-VIS'] = 1.0
            if vl['TEAM-NAME'] == vn['NEXT-HOME-TEAM']:
                ls['VIS-NEXT-HOME'] = 1.0
                ls['VIS-NEXT-VIS'] = 0.0
            if vl['TEAM-NAME'] == vn['NEXT-VISITING-TEAM']:
                ls['VIS-NEXT-HOME'] = 0.0
                ls['VIS-NEXT-VIS'] = 1.0
            else:
                ls[f'HOME-{k}'] = v
                ls[f'VIS-{k}'] = vn[k]

defeat_prob = pd.read_csv('./templates/team_defeat_problem.csv')
defeat_sol = pd.read_csv('./templates/team_defeat_solution.csv')

## Generate the text for defeat
case_base_sim_ftrs = defeat_prob['sim_features'].tolist()
case_base_sim_ftrs = [json.loads(i) for i in case_base_sim_ftrs]
case_base_sim_ftrs_keys = list(case_base_sim_ftrs[0].keys())

solution_templates = defeat_sol['templates'].tolist()
case_base_sim_ftrs_arr = np.array([list(i.values()) for i in case_base_sim_ftrs])

# Extract the sim_ftrs for target problem
target_prob_sim_ftrs = {}
for k, v in ls.items():
    if k in case_base_sim_ftrs_keys:
        target_prob_sim_ftrs[k] = v
target_prob_sim_ftrs_arr = np.array(list(target_prob_sim_ftrs.values()))

# print(case_base_sim_ftrs_arr, target_prob_sim_ftrs_arr)

dists = euclidean_distances(case_base_sim_ftrs_arr, [target_prob_sim_ftrs_arr])
dists_1d = dists.ravel()
dists_arg = np.argsort(dists_1d)[:5]

# proposed solutions
proposed_solutions = []
for i in dists_arg:
    tmpl = solution_templates[i]
    new_str = ""
    for tok in tmpl.split(' '):
        if tok.isupper() and tok in list(ls.keys()):
            new_str += f"{str(ls[tok])} "
        else:
            new_str += f"{tok} "
    proposed_solutions.append(new_str)

# TODO: Use LM-Scoring
print("\n\nThis game")
print(proposed_solutions[0])
# print(ls['VIS-NEXT-HOME'], ls['VIS-NEXT-VIS'])

if next_game_exist:
    next_game_prob = pd.read_csv('./templates/team_next-game_problem.csv')
    next_game_sol = pd.read_csv('./templates/team_next-game_solution.csv')

    for generating_for in ['HOME', 'VIS']:
        # generating_for = 'HOME' # ['HOME', 'VIS'] HOME or VIS - for which team you're generating

        ## Generate the text for defeat
        case_base_sim_ftrs = next_game_prob['sim_features'].tolist()
        case_base_sim_ftrs = [json.loads(i) for i in case_base_sim_ftrs]

        solution_templates = next_game_sol['templates'].tolist()

        # only take keys for home when generating for home
        if generating_for == 'HOME':
            case_base_sim_ftrs_new = []
            for item in case_base_sim_ftrs:
                tmp = {}
                for k, v in item.items():
                    if k not in vis_next_game_sim_ftrs_keys:
                        tmp[k] = v
                case_base_sim_ftrs_new.append(tmp)
        else:
            case_base_sim_ftrs_new = []
            for item in case_base_sim_ftrs:
                tmp = {}
                for k, v in item.items():
                    if k not in home_next_game_sim_ftrs_keys:
                        tmp[k] = v
                case_base_sim_ftrs_new.append(tmp)
        case_base_sim_ftrs_arr = np.array([list(i.values()) for i in case_base_sim_ftrs_new])
        # print(case_base_sim_ftrs_new[0])

        # Extract the sim_ftrs for target problem
        target_prob_sim_ftrs = {}
        for k, v in ls.items():
            if k in case_base_sim_ftrs_new[0].keys():
                target_prob_sim_ftrs[k] = v
        target_prob_sim_ftrs_arr = np.array(list(target_prob_sim_ftrs.values()))
        # print(target_prob_sim_ftrs_arr.shape, case_base_sim_ftrs_arr.shape)

        dists = euclidean_distances(case_base_sim_ftrs_arr, [target_prob_sim_ftrs_arr])
        dists_1d = dists.ravel()
        dists_arg = np.argsort(dists_1d)[:5]

        # proposed solutions
        proposed_solutions = []
        for i in dists_arg:
            tmpl = solution_templates[i]
            new_str = ""
            # print(tmpl)
            for tok in tmpl.split(' '):
                if generating_for == 'HOME':
                    new_tok = f'HOME-{tok}'
                else:
                    new_tok = f'VIS-{tok}'
                if new_tok in list(ls.keys()):
                    new_str += f"{str(ls[new_tok])} "
                else:
                    new_str += f"{new_tok.split('-')[1]} "
            proposed_solutions.append(new_str)

        print(f"\n\nNext game:\t{generating_for}")
        print(proposed_solutions[0], solution_templates[dists_arg[0]])

print(ls)

# # TODO: Apply LaserTagger here
# # final_sol = ""
# # if next_game_exist:
# #     prop_sols
# # else:
# #     final_sol = prop_sols[0]