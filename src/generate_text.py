import json, pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from text2num import text2num, NumberException
from imp_players_utility import ImportantPlayers 
from misc import MiscTasks
from template_utility import TemplateExtractionUtility

mt = MiscTasks()
teu = TemplateExtractionUtility()
player_clusters = mt.player_clusts
team_clusters = mt.team_clusts
nick_names = mt.nick_names
all_atts = json.load(open('./data/atts.json', 'r'))

def generating_player_text_from_templates(js, game_idx, tokenizer):
    # js = json.load(open(f'./data/jsons/2018_w_opp.json', 'r'))
    # game_idx = 11

    imp_ps = ImportantPlayers()

    imp_players_stats = {}
    """
    # this is using NBA efficiency formula
    home_imp_players, vis_imp_players = imp_ps.select_imp_player_on_eff(js[game_idx]['box_score'])
    imp_players_stats.update(get_player_score(list(home_imp_players.keys())[:3], js[game_idx]))
    imp_players_stats.update(get_player_score(list(vis_imp_players.keys())[:3], js[game_idx]))
    """

    # this one's using trained model
    pkl_filename = f"./data/imp_players/tpot_best_model_imp_player_classifier.pkl"    
    with open(pkl_filename, 'rb') as file:
        clf = pickle.load(file)
    home_imp_players, vis_imp_players = imp_ps.select_imp_player_by_model(clf, js[game_idx])
    if len(home_imp_players) > 3:
        imp_players_stats.update(teu.get_player_score(home_imp_players[:3], js[game_idx]))
    else:
        imp_players_stats.update(teu.get_player_score(home_imp_players, js[game_idx]))
    if len(vis_imp_players) > 3:
        imp_players_stats.update(teu.get_player_score(vis_imp_players[:3], js[game_idx]))
    else:
        imp_players_stats.update(teu.get_player_score(vis_imp_players, js[game_idx]))

    # print(imp_players_stats)
    # print(len(imp_players_stats))

    # ftr_weights = np.array(list(json.load(open('./data/align_data/player/feature_weights.json', 'r')).values()))
    ftr_weights = np.array(list(json.load(open('./data/imp_players/ftr_weights_info_gain.json', 'r')).values()))
    # print(ftr_weights.shape)
    scaler_filename = f"./data/align_data/player/data_scaler.pkl"
    with open(scaler_filename, 'rb') as file:
        scaler_model = pickle.load(file)


    cb_player_stats_problem = pd.read_csv(f'./data/case_base/player_stats_problem.csv')
    cb_player_stats_solution = pd.read_csv(f'./data/case_base/player_stats_solution.csv')

    case_base_sim_ftrs = cb_player_stats_problem['sim_features'].tolist()
    case_base_sim_ftrs = [json.loads(i) for i in case_base_sim_ftrs]
    case_base_sim_ftrs_arr = np.array([list(i.values()) for i in case_base_sim_ftrs])
    # apply scaling
    case_base_sim_ftrs_arr = scaler_model.transform(case_base_sim_ftrs_arr)
    # apply feature weights
    case_base_sim_ftrs_arr = np.multiply(case_base_sim_ftrs_arr, ftr_weights)
    # print(case_base_sim_ftrs_arr.shape)

    solution_templates = cb_player_stats_solution['templates'].tolist()

    # print(case_base_sim_ftrs_arr.shape)
    all_players_proposed_solutions = {}

    for player, player_stats in imp_players_stats.items():
        # print(player, player_stats)
        player_sim_ftrs = {}
        for k, v in player_stats.items():
            if not isinstance(v, str):
                player_sim_ftrs[k] = v
        target_problem_sim_ftrs_arr = np.array(list(player_sim_ftrs.values()))
        # apply data scaling
        target_problem_sim_ftrs_arr = scaler_model.transform([target_problem_sim_ftrs_arr])
        # apply feature weights
        target_problem_sim_ftrs_arr = np.multiply(target_problem_sim_ftrs_arr, ftr_weights)
        # print(target_problem_sim_ftrs_arr.shape)

        # Now calculate the simialrity between case-base and each player
        dists = euclidean_distances(case_base_sim_ftrs_arr, target_problem_sim_ftrs_arr)
        dists_1d = dists.ravel()
        dists_arg = np.argsort(dists_1d)[:5]

        # proposed solutions
        proposed_solutions = {}
        for i in dists_arg:
            tmpl = solution_templates[i]
            new_str = ""
            for tok in tmpl.split(' '):
                if tok in list(player_stats.keys()):
                    new_str += f"{str(player_stats[tok])} "
                else:
                    new_str += f"{tok} "
            # applying lm-scoring here
            text = new_str.replace("_", " ")
            tokens_tensor = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
            score = mt.score_sent(tokens_tensor)
            proposed_solutions[new_str] = score

        proposed_solutions_sorted = {k: v for k, v in sorted(proposed_solutions.items(), key=lambda item: item[1])}
        for idx, (k, v) in enumerate(proposed_solutions_sorted.items()):
            if idx == 0:
                all_players_proposed_solutions[player] = k

        # for idx, (k, v) in enumerate(proposed_solutions.items()):
        #     if idx == 0:
        #         all_players_proposed_solutions[player] = k

    return all_players_proposed_solutions


def generating_team_text_from_templates(test_json, game_idx, tokenizer):

    score_dict = test_json[game_idx]

    ftr_weights = np.array(list(json.load(open('./data/align_data/team/feature_weights.json', 'r')).values()))
    scaler_filename = f"./data/align_data/team/data_scaler.pkl"
    with open(scaler_filename, 'rb') as file:
        scaler_model = pickle.load(file)

    cb_teams_stats_problem = pd.read_csv(f'./data/case_base/team_stats_problem.csv')
    cb_teams_stats_solution = pd.read_csv(f'./data/case_base/team_stats_solution.csv')

    case_base_sim_ftrs = cb_teams_stats_problem['sim_features'].tolist()
    case_base_sim_ftrs = [json.loads(i) for i in case_base_sim_ftrs]
    case_base_sim_ftrs_arr = np.array([list(i.values()) for i in case_base_sim_ftrs])
    case_base_sim_ftrs_arr = scaler_model.transform(case_base_sim_ftrs_arr)
    # case_base_sim_ftrs_arr = np.multiply(case_base_sim_ftrs_arr, ftr_weights)

    solution_templates = cb_teams_stats_solution['templates'].tolist()

    target_problem_stats, target_problem_sim_ftrs = teu.get_team_score(score_dict)
    target_problem_sim_ftrs_arr = np.array(list(target_problem_sim_ftrs.values()))
    target_problem_sim_ftrs_arr = scaler_model.transform([target_problem_sim_ftrs_arr])
    # target_problem_sim_ftrs_arr = np.multiply(target_problem_sim_ftrs_arr, ftr_weights)

    dists = euclidean_distances(case_base_sim_ftrs_arr, target_problem_sim_ftrs_arr)
    dists_1d = dists.ravel()
    dists_arg = np.argsort(dists_1d)[:5]

    # proposed solutions
    proposed_solutions = {}
    for i in dists_arg:
        tmpl = solution_templates[i]
        new_str = ""
        for tok in tmpl.split(' '):
            if tok in list(target_problem_stats.keys()):
                new_str += f"{str(target_problem_stats[tok])} "
            else:
                new_str += f"{tok} "
        
        # applying lm-scoring here
        text = new_str.replace("_", " ")
        tokens_tensor = tokenizer.encode(text, add_special_tokens=False, return_tensors="pt")
        score = mt.score_sent(tokens_tensor)
        proposed_solutions[new_str] = score

    proposed_solutions_sorted = {k: v for k, v in sorted(proposed_solutions.items(), key=lambda item: item[1])}
    team_stats_final_sol = {}
    for idx, (k, v) in enumerate(proposed_solutions_sorted.items()):
        if idx == 0:
            team_stats_final_sol[f'team{idx+1}'] = k

    # team_stats_final_sol = {}
    # for idx, (k, v) in enumerate(proposed_solutions.items()):
    #     if idx == 0:
    #         team_stats_final_sol[f'team{idx+1}'] = k

    return team_stats_final_sol

# print(generating_text_from_templates())
# generating_player_text_from_templates(11)

# sent = "Nerlens Noel had a double - double , putting up 10 points on 5 - of - 12 shooting and 11 rebounds in 37 minutes ."
# stats = json.loads(pd.rea
# d_csv('./data/case_base/player_stats_problem.csv')['sim_features'][26])
# # stats = json.loads(
# #     "{""STARTER"": 0, ""PTS"": 10, ""FGM"": 5, ""FGA"": 12, ""FG_PCT"": 42, ""FG3M"": 0, ""FG3A"": 0, ""FG3_PCT"": 0, ""FTM"": 0, ""FTA"": 0, ""FT_PCT"": 0, ""OREB"": 5, ""DREB"": 6, ""REB"": 11, ""AST"": 1, ""TO"": 0, ""STL"": 6, ""BLK"": 0, ""PF"": 5, ""MIN"": 37, ""IS_HOME"": 0, ""PTS_AVG"": 8, ""STL_AVG"": 1, ""BLK_AVG"": 1, ""REB_AVG"": 7, ""AST_AVG"": 1, ""DOUBLE_DOUBLE"": 0, ""DD_AVG"": 10, ""IS_LEADER"": 0}"
# # )
# stats.update({"FIRST_NAME": "Nerlens", "SECOND_NAME": "Noel"})
# print(stats)
# t = ext_player_temp_from_sent(sent, {1: stats})
# print(t)