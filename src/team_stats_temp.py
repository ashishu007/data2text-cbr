"""
Extract teams' related templates
1. in case of teams, can easily take stats from both the teams and use them for simialrity
2. doesn't matter if the sentence is just about one team or two
3. also, don't need next-game info, just current one will do
"""

import json
import pandas as pd
import numpy as np
from text2num import text2num, NumberException
from misc import MiscTasks
from sklearn.metrics.pairwise import euclidean_distances

mt = MiscTasks()

team_clusters = mt.team_clusts
# team_clusters = ['Y', 'F']

nick_names = {"Sixers": "76ers", "Cavs": "Cavaliers", "T'wolves": "Timberwolves", "Blazers": "Trail_Blazers", "OKC": "Oklahoma_City"}
all_atts = json.load(open('./data/atts.json', 'r'))

def get_team_score(score_dict):

    team_stats, sim_ftrs = {}, {}

    for key in all_atts['line-score keys']:
        team_stats[f"HOME-{key}"] = score_dict['home_line'][key]
        team_stats[f"VIS-{key}"] = score_dict['vis_line'][key]

    for key in all_atts['line-score sim_ftrs keys']:
        if key not in ['WINNER', 'STREAK-TYPE']:
            sim_ftrs[f"HOME-{key}"] = int(score_dict['home_line'][key])
            sim_ftrs[f"VIS-{key}"] = int(score_dict['vis_line'][key])
        elif key == 'WINNER':
            sim_ftrs[f"HOME-{key}"] = 1 if score_dict['home_line']['WINNER'] == 'yes' else 0
            sim_ftrs[f"VIS-{key}"] = 1 if score_dict['vis_line']['WINNER'] == 'yes' else 0
        elif key == 'STREAK-TYPE':
            sim_ftrs[f"HOME-{key}"] = 1 if score_dict['home_line']['STREAK-TYPE'] == 'W' else 0
            sim_ftrs[f"VIS-{key}"] = 1 if score_dict['vis_line']['STREAK-TYPE'] == 'W' else 0
    
    return team_stats, sim_ftrs

def ext_team_temp_from_sent(sent, stats):
    new_toks, used_atts = [], []
    for tok in sent.split(' '):
        found = False
        key = ''
        for k, v in stats.items():
            if tok == str(v):
                found = True
                key = k
                used_atts.append(key)
        if found:
            new_toks.append(key)
        else:
            new_toks.append(tok)
    template = ' '.join(new_toks)
    return template, used_atts

def extracting_team_stats_templates_from_texts():
    jsons = {}
    for season in [2014, 2015, 2016]:
        js1 = json.load(open(f'./data/jsons/{season}_w_opp.json', 'r'))
        jsons[season] = js1

    # df = pd.read_csv('./data/clusters/all_clusters.csv')
    df = pd.read_csv(mt.cluster_path)
    df1 = df.loc[df['clust'].isin(team_clusters)]

    problem_side = {"sentences": [], "sim_features": []}
    solution_side = {"templates": [], "used_attributes": []}

    for idx, row in df1.iterrows():
        if idx % 1000 == 0:
            print(idx)

        # if idx < 100:
        sent = row['sent_wo_ord']
        game_idx = row['game_idx']
        js = jsons[row['season']]
        score_dict = js[game_idx]
        
        new_toks = []
        for tok in sent.split(' '):
            try:
                t = text2num(tok)
                new_toks.append(str(t))
            except NumberException:
                if tok in nick_names:
                    new_toks.append(nick_names[tok])
                else:
                    new_toks.append(tok)
        new_sent = ' '.join(new_toks)

        teams_stats, sim_ftrs = get_team_score(score_dict)
        template, used_atts = ext_team_temp_from_sent(new_sent, teams_stats)

        problem_side["sentences"].append(new_sent)
        problem_side["sim_features"].append(json.dumps(sim_ftrs))
        solution_side["templates"].append(template)
        solution_side["used_attributes"].append(used_atts)

    dfp = pd.DataFrame(problem_side)
    dfs = pd.DataFrame(solution_side)
    print(dfs.shape)
    print(dfp.shape)
    dfp.to_csv(f'./data/case_base/team_stats_problem.csv', index=0)
    dfs.to_csv(f'./data/case_base/team_stats_solution.csv', index=0)

def generating_team_text_from_templates(test_json, game_idx):
    # test_json = json.load(open(f'./data/jsons/2018_w_opp.json', 'r'))
    # game_idx = 11

    # team_stats_final_sol = {}

    score_dict = test_json[game_idx]

    target_problem_stats, target_problem_sim_ftrs = get_team_score(score_dict)
    target_problem_sim_ftrs_arr = np.array(list(target_problem_sim_ftrs.values()))

    cb_teams_stats_problem = pd.read_csv(f'./data/case_base/team_stats_problem.csv')
    cb_teams_stats_solution = pd.read_csv(f'./data/case_base/team_stats_solution.csv')

    case_base_sim_ftrs = cb_teams_stats_problem['sim_features'].tolist()
    case_base_sim_ftrs = [json.loads(i) for i in case_base_sim_ftrs]
    case_base_sim_ftrs_arr = np.array([list(i.values()) for i in case_base_sim_ftrs])

    solution_templates = cb_teams_stats_solution['templates'].tolist()

    dists = euclidean_distances(case_base_sim_ftrs_arr, [target_problem_sim_ftrs_arr])
    dists_1d = dists.ravel()
    dists_arg = np.argsort(dists_1d)[:5]

    # proposed solutions
    proposed_solutions = []
    for i in dists_arg:
        tmpl = solution_templates[i]
        new_str = ""
        for tok in tmpl.split(' '):
            if tok in list(target_problem_stats.keys()):
                new_str += f"{str(target_problem_stats[tok])} "
            else:
                new_str += f"{tok} "
        proposed_solutions.append(new_str)

    # print(proposed_solutions)
    # print(target_problem_stats)
    team_stats_final_sol = {"team1": proposed_solutions[0], "team2": proposed_solutions[1], "team3": proposed_solutions[2]}
    return team_stats_final_sol


# test_json = json.load(open(f'./data/jsons/2018_w_opp.json', 'r'))
# game_idx = 11
# s = generating_team_text_from_templates(test_json, 11)
# print(s)