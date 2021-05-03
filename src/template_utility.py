import json
from text2num import text2num
from nltk import word_tokenize

class TemplateExtractionUtility:
    def __init__(self):
        self.all_atts = json.load(open('./data/atts.json', 'r'))


    def get_all_ents(self, score_dict):
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

    def extract_entities(self, all_ents, sent):
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

    def str_nums_to_int_nums(self, sent):
        new_toks = []
        for tok in word_tokenize(sent):
            try:
                t = text2num(tok)
                new_toks.append(str(t))
            except:
                new_toks.append(tok)
        return ' '.join(new_toks)

    def ext_temp_from_sent(self, sent, stats):
        new_toks, used_atts = [], []
        new_sent = self.str_nums_to_int_nums(sent)
        for tok in new_sent.split(' '):
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

    def get_player_score(self, player_ents, score_dict):
        
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
                if k in self.all_atts['box-score sim_ftrs keys']:
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

    def get_team_score(self, score_dict):

        team_stats, sim_ftrs = {}, {}

        for key in self.all_atts['line-score keys']:
            team_stats[f"HOME-{key}"] = score_dict['home_line'][key]
            team_stats[f"VIS-{key}"] = score_dict['vis_line'][key]

        for key in self.all_atts['line-score sim_ftrs keys']:
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
