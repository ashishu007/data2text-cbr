"""
Extract players' related templates
1. check if there's one player or two
2. add num_entities as feature
3. entity matching

Process: 
0. add players' team names to the box-score
1. identify which player/entity it is taking about
2. get the stats of that player/entity
3. replace the matching tokens with their feature_name
4. similarity_ftrs will be players' stats

add leader - yes/no (is_leader); team's name to player stats (own_team_name/opponent_team_name)
add conceptId as feature as well (cluster identifier - clustid)

Impact of clusters:
I'm hoping with weights learned for the features, they would be able to pick important things.
For example, if a player has scored d-d, a template with d-d cluster will be picked.
I can't add the clusterId as similarity feature because this woudn't be available during testing.
"""

import json, pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from text2num import text2num, NumberException
from select_imp_players import ImportantPlayers #select_imp_player_on_eff
from misc import MiscTasks

mt = MiscTasks()
player_clusters = mt.player_clusts
nick_names = mt.nick_names
all_atts = json.load(open('./data/atts.json', 'r'))

def get_all_ents(score_dict):
    players = set()
    teams = set()

    players.update(score_dict['box_score']['PLAYER_NAME'].values())
    players.update(score_dict['box_score']['FIRST_NAME'].values())
    players.update(score_dict['box_score']['SECOND_NAME'].values())

    teams.add(score_dict['home_name'])
    teams.add(score_dict['vis_name'])
    
    teams.add(score_dict['home_city'])
    teams.add(score_dict['vis_city'])
    
    teams.add(f"{score_dict['home_city']} {score_dict['home_name']}")
    teams.add(f"{score_dict['vis_city']} {score_dict['vis_name']}")

    all_ents = teams | players

    return all_ents, teams, players

def extract_entities(all_ents, sent):
    toks = sent.split(' ')
    sent_ents = []
    i = 0
    while i < len(toks):
        if toks[i] in all_ents:
            j = 1
            while i+j <= len(toks) and " ".join(toks[i:i+j]) in all_ents:
                j += 1
            sent_ents.append(" ".join(toks[i:i+j-1]))
            i += j-1
        else:
            i += 1
    return list(set(sent_ents))

def get_player_score(player_ents, score_dict):
    
    full_names = list(score_dict['box_score']['PLAYER_NAME'].values())
    first_names = list(score_dict['box_score']['FIRST_NAME'].values())
    last_names = list(score_dict['box_score']['SECOND_NAME'].values())

    all_player_stats = {}
    for player in player_ents:

        player_idx = -1
        if player in full_names:
            for k, v in score_dict['box_score']['PLAYER_NAME'].items():
                if player == v: player_idx = k
        elif player in first_names:
            for k, v in score_dict['box_score']['FIRST_NAME'].items():
                if player == v: player_idx = k
        elif player in last_names:
            for k, v in score_dict['box_score']['SECOND_NAME'].items():
                if player == v: player_idx = k

        player_stats = {}
        for k, v in score_dict['box_score'].items():
            if k in all_atts['box-score sim_ftrs keys']:
                if k not in ['STARTER', 'IS_HOME', 'DOUBLE_DOUBLE']:
                    val = int(v[str(player_idx)]) if v[str(player_idx)] != 'N/A' else 0
                    player_stats[k] = val
                else:
                    val = 1 if v == 'yes' else 0
                    player_stats[k] = val
        
        if score_dict['box_score']['IS_HOME'][player_idx] == 'yes':
            is_leader = 1 if f"{score_dict['home_line']['LEADER_FIRST_NAME']} {score_dict['home_line']['LEADER_SECOND_NAME']}" == score_dict['box_score']['PLAYER_NAME'][player_idx] else 0
        else:
            is_leader = 1 if f"{score_dict['vis_line']['LEADER_FIRST_NAME']} {score_dict['vis_line']['LEADER_SECOND_NAME']}" == score_dict['box_score']['PLAYER_NAME'][player_idx] else 0
        player_stats['IS_LEADER'] = is_leader

        player_stats['PLAYER-TEAM-NAME'] = score_dict['home_name'] if score_dict['box_score']['IS_HOME'][player_idx] == 'yes' else score_dict['vis_name']
        player_stats['PLAYER-TEAM-PLACE'] = score_dict['home_city'] if score_dict['box_score']['IS_HOME'][player_idx] == 'yes' else score_dict['vis_city']

        player_stats['PLAYER-OPPONENT-TEAM-NAME'] = score_dict['home_name'] if score_dict['box_score']['IS_HOME'][player_idx] == 'no' else score_dict['vis_name']
        player_stats['PLAYER-OPPONENT-TEAM-PLACE'] = score_dict['home_city'] if score_dict['box_score']['IS_HOME'][player_idx] == 'no' else score_dict['vis_city']

        player_stats['PLAYER_NAME'] = score_dict['box_score']['PLAYER_NAME'][player_idx]
        player_stats['FIRST_NAME'] = score_dict['box_score']['FIRST_NAME'][player_idx]
        player_stats['SECOND_NAME'] = score_dict['box_score']['SECOND_NAME'][player_idx]

        all_player_stats[player] = player_stats
        # print(player_stats)
    # print(all_player_stats)
    return all_player_stats

def ext_player_temp_from_sent(sent, stats):
    player_stats = list(stats.values())[0]
    new_toks, used_atts = [], []
    for tok in sent.split(' '):
        found = False
        key = ''
        for k, v in player_stats.items():
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

def extracting_player_stats_templates_from_texts():
    jsons = {}
    for season in [2014, 2015, 2016]:
        js1 = json.load(open(f'./data/jsons/{season}_w_opp.json', 'r'))
        jsons[season] = js1

    team_names = json.load(open(f'./data/team_names.json', 'r'))['Team Names']

    # df = pd.read_csv('./data/clusters/all_clusters.csv')
    df = pd.read_csv(mt.cluster_path)
    df1 = df.loc[df['clust'].isin(player_clusters)]

    problem_side = {"sentences": [], "sim_features": []}
    solution_side = {"templates": [], "used_attributes": []}

    for idx, row in df1.iterrows():
        if idx % 1000 == 0:
            print(idx)

        sent = row['sent_wo_ord']
        game_idx = row['game_idx']
        js = jsons[row['season']]
        score_dict = js[game_idx]
        all_ents, team_ents, player_ents = get_all_ents(score_dict)

        new_toks = []
        for tok in sent.split(' '):
            try:
                t = text2num(tok)
                new_toks.append(str(t))
            except:
                if tok in nick_names:
                    new_toks.append(nick_names[tok])
                elif f'{tok}s' in nick_names:
                    new_toks.append(nick_names[f'{tok}s'])
                elif f'{tok}s' in team_names:
                    new_toks.append(f'{tok}s')
                else:
                    new_toks.append(tok)
        new_sent = ' '.join(new_toks)

        players = extract_entities(player_ents, new_sent)

        player_ent_found = True if len(players) > 0 else False
        if player_ent_found:
            player_stats = get_player_score(players, score_dict)
            if len(player_stats) == 1:
                template, used_atts = ext_player_temp_from_sent(new_sent, player_stats)
                sim_ftrs = {}
                for k, v in list(player_stats.values())[0].items():
                    if not isinstance(v, str):
                        sim_ftrs[k] = v
                problem_side["sentences"].append(new_sent)
                problem_side["sim_features"].append(json.dumps(sim_ftrs))
                solution_side["templates"].append(template)
                solution_side["used_attributes"].append(used_atts)


    dfp = pd.DataFrame(problem_side)
    dfs = pd.DataFrame(solution_side)
    print(dfs.shape)
    print(dfp.shape)
    dfp.to_csv(f'./data/case_base/player_stats_problem.csv', index=0)
    dfs.to_csv(f'./data/case_base/player_stats_solution.csv', index=0)

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
        imp_players_stats.update(get_player_score(home_imp_players[:3], js[game_idx]))
    else:
        imp_players_stats.update(get_player_score(home_imp_players, js[game_idx]))
    if len(vis_imp_players) > 3:
        imp_players_stats.update(get_player_score(vis_imp_players[:3], js[game_idx]))
    else:
        imp_players_stats.update(get_player_score(vis_imp_players, js[game_idx]))

    # print(imp_players_stats)
    # print(len(imp_players_stats))

    ftr_weights = np.array(list(json.load(open('./data/align_data/player/feature_weights.json', 'r')).values()))
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

    return all_players_proposed_solutions

# print(generating_text_from_templates())
# generating_player_text_from_templates(11)