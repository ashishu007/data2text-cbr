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
"""

import json
import pandas as pd
from text2num import text2num, NumberException

player_clusters = ['A', 'D', 'E', 'G', 'H', 'I', 'N', 'O', 'R', 'T', 'V']

nick_names = {"Sixers": "76ers", "Cavs": "Cavaliers", "T'wolves": "Timberwolves", "Blazers": "Trail_Blazers"}
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

def extracting_templates_from_texts():
    jsons = {}
    for season in [2014, 2015, 2016]:
        js1 = json.load(open(f'./data/jsons/{season}_w_opp.json', 'r'))
        jsons[season] = js1

    df = pd.read_csv('./data/clusters/all_clusters.csv')
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
            except NumberException:
                if tok in nick_names:
                    new_toks.append(nick_names[tok])
                else:
                    new_toks.append(tok)
        new_sent = ' '.join(new_toks)

        players = extract_entities(player_ents, new_sent)
        teams = extract_entities(team_ents, new_sent)

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

def generating_text_from_templates():
    pass