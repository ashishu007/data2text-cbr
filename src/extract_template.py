import json
import pandas as pd
from template_utility import TemplateExtractionUtility
from misc import MiscTasks

mt = MiscTasks()
teu = TemplateExtractionUtility()
player_clusters = mt.player_clusts
team_clusters = mt.team_clusts
nick_names = mt.nick_names

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
        all_ents, team_ents, player_ents = teu.get_all_ents(score_dict)

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

        players = teu.extract_entities(player_ents, new_sent)

        player_ent_found = True if len(players) > 0 else False
        if player_ent_found:
            player_stats = teu.get_player_score(players, score_dict)
            if len(player_stats) == 1:
                template, used_atts = teu.ext_temp_from_sent(new_sent, list(player_stats.values())[0])
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

def extracting_team_stats_templates_from_texts():
    jsons = {}
    for season in [2014, 2015, 2016]:
        js1 = json.load(open(f'./data/jsons/{season}_w_opp.json', 'r'))
        jsons[season] = js1
    
    team_names = json.load(open(f'./data/team_names.json', 'r'))['Team Names']

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
        all_ents, team_ents, player_ents = teu.get_all_ents(score_dict)

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

        teams = teu.extract_entities(team_ents, new_sent)
        team_ents_found = True if len(teams) > 0 else False
        # print(teams)

        if team_ents_found and len(teams) > 1:
            teams_stats, sim_ftrs = teu.get_team_score(score_dict)
            template, used_atts = teu.ext_temp_from_sent(new_sent, teams_stats)

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

print(f'extracting players stats templates')
extracting_player_stats_templates_from_texts()

print(f'extracting teams stats templates')
extracting_team_stats_templates_from_texts()

print(f"all done")