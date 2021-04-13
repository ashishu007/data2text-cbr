import json
import pandas as pd
from text2num import text2num, NumberException

sim_ftrs_keys = ["TEAM-PTS_QTR1", "TEAM-PTS_QTR2", "TEAM-PTS_QTR3", "TEAM-PTS_QTR4", "TEAM-PTS", "TEAM-FGM", "TEAM-FGA", "TEAM-FG_PCT",
		        "TEAM-FG3M", "TEAM-FG3A", "TEAM-FG3_PCT", "TEAM-FTM", "TEAM-FTA", "TEAM-FT_PCT", "TEAM-REB", "TEAM-AST", "TEAM-TOV", 
        		"TEAM-WINS", "TEAM-LOSSES", "WINNER", "TEAM-STANDING", "STREAK-TYPE", "STREAK-COUNT"]
next_game_sim_ftrs_keys = ["HOME-NEXT-HOME", "HOME-NEXT-VIS", "VIS-NEXT-HOME", "VIS-NEXT-VIS"]

all_atts = json.load(open('atts.json', 'r'))

nick_names = {"Sixers": "76ers", "Cavs": "Cavaliers", "T'wolves": "Timberwolves", "Blazers": "Trail_Blazers"}

# 1. Extract Team A defeated B types
def team_templates(df1, jsons, cat='defeat'):
    problem_side = {"sentences": [], "sim_features": []}
    solution_side = {"templates": [], "used_attributes": []}

    for idx, row in df1.iterrows():
        if idx % 1000 == 0:
            print(idx)

        try:
            sent = row['sent']
            game_idx = row['game_idx']
            js = jsons[row['season']]

            hl = js[game_idx]['home_line']
            vl = js[game_idx]['vis_line']
            g = js[game_idx]['game']
            hn = js[game_idx]['home_next_game']
            vn = js[game_idx]['vis_next_game']

            ls, sim_ftrs = {}, {} # sim_ftrs are similarity features used for measuring the similarity during generation 
            for k, v in hl.items():
                if k in all_atts['line-score keys']:
                    ls[f'HOME-{k}'] = v
                    ls[f'VIS-{k}'] = vl[k]
                # if cat == 'defeat':
                if k in sim_ftrs_keys:
                    if k == 'WINNER':
                        sim_ftrs[f'HOME-{k}'] = 0.0 if v == 'no' else 1.0
                        sim_ftrs[f'VIS-{k}'] = 0.0 if vl[k] == 'no' else 1.0
                    elif k == 'STREAK-TYPE':
                        sim_ftrs[f'HOME-{k}'] = 0.0 if v == 'L' else 1.0
                        sim_ftrs[f'VIS-{k}'] = 0.0 if vl[k] == 'L' else 1.0
                    else:
                        sim_ftrs[f"HOME-{k}"] = float(v)
                        sim_ftrs[f"VIS-{k}"] = float(vl[k])
                if cat == 'next-game':
                    # if next-game simialrity features should be H/V's next game is H or V
                    if hl['TEAM-NAME'] == hn['NEXT-HOME-TEAM']:
                        sim_ftrs['HOME-NEXT-HOME'] = 1.0
                        sim_ftrs['HOME-NEXT-VIS'] = 0.0
                    if hl['TEAM-NAME'] == hn['NEXT-VISITING-TEAM']:
                        sim_ftrs['HOME-NEXT-HOME'] = 0.0
                        sim_ftrs['HOME-NEXT-VIS'] = 1.0
                    if vl['TEAM-NAME'] == vn['NEXT-HOME-TEAM']:
                        sim_ftrs['VIS-NEXT-HOME'] = 1.0
                        sim_ftrs['VIS-NEXT-VIS'] = 0.0
                    if vl['TEAM-NAME'] == vn['NEXT-VISITING-TEAM']:
                        sim_ftrs['VIS-NEXT-HOME'] = 0.0
                        sim_ftrs['VIS-NEXT-VIS'] = 1.0

            for k, v in g.items():
                if k in all_atts['game keys']:
                    ls[f'{k}'] = v
            for k, v in hn.items():
                if k in all_atts['next-game keys']:
                    ls[f'HOME-{k}'] = v
                    ls[f'VIS-{k}'] = vn[k]

            # change the possible string numbers to int numbers
            new_sent = []
            for i in sent.split(' '):
                try:
                    t = text2num(i)
                    new_sent.append(str(t))
                except NumberException:
                    if i in nick_names:
                        new_sent.append(str(nick_names[i]))
                    else:
                        new_sent.append(str(i))

            new_sent = ' '.join(new_sent)

            # replace any occurence of token match with line-score to its key
            new_toks, used_atts = [], []
            for tok in new_sent.split(' '):
                found = False
                key = ''

                for k, v in ls.items():
                    if tok == str(v):
                        found = True
                        key = k if cat == 'defeat' else '-'.join(k.split('-')[1:])
                        used_atts.append(key)
                        break
                if found:
                    new_toks.append(key)
                else:
                    new_toks.append(tok)
            template = ' '.join(new_toks)

            problem_side["sentences"].append(new_sent)
            problem_side["sim_features"].append(json.dumps(sim_ftrs))
            solution_side["templates"].append(template)
            solution_side["used_attributes"].append(used_atts)
            
        except:
            print('an error')

    dfp = pd.DataFrame(problem_side)
    dfs = pd.DataFrame(solution_side)

    print(dfs.shape)
    print(dfp.shape)

    dfp.to_csv(f'./templates/team_{cat}_problem.csv', index=0)
    dfs.to_csv(f'./templates/team_{cat}_solution.csv', index=0)


jsons = {}
for season in [2014, 2015, 2016]:
    js1 = json.load(open(f'./data/jsons/{season}_new_atts_w_stand_streak.json', 'r'))
    jsons[season] = js1

df = pd.read_csv('./data/clusters/sents_from_code_w_clusts.csv')

for cat in ['next-game', 'defeat']:
    print(cat)

    if cat == 'defeat':
        df2 = df.loc[df['clust'].isin(['B', 'C'])]
    elif cat == 'next-game':
        df2 = df.loc[df['clust'].isin(['J'])]

    print(df2.shape)
    team_templates(df2, jsons, cat)

