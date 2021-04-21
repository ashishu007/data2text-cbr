"""
Rule to generate defeat and next-game sentences
"""
import json
import pandas as pd
import numpy as np
from datetime import datetime

class RulesForGeneration:
    def __init__(self):
        # next game templates
        self.ngts = {
            "ngt1": "The TEAM-NAME will stay home to host NEXT-OPPONENT-TEAM on NEXT-DAYNAME .",
            "ngt2": "The TEAM-NAME will look to continue their winning streak when they host NEXT-OPPONENT-TEAM on NEXT-DAYNAME .", 
            "ngt3": "Next , the TEAM-NAME will head to NEXT-OPPONENT-TEAM-PLACE to face NEXT-OPPONENT-TEAM on NEXT-DAYNAME .", 
            "ngt4": "The TEAM-NAME now head to NEXT-OPPONENT-TEAM-PLACE for a NEXT-DAYNAME night showdown versus the NEXT-OPPONENT-TEAM .", 
            "ngt5": "The TEAM-NAME will look to bounce back when they host NEXT-OPPONENT-TEAM-PLACE NEXT-OPPONENT-TEAM on NEXT-DAYNAME .", 
            "ngt6": "The TEAM-PLACE TEAM-NAME will return home to face the NEXT-OPPONENT-TEAM on NEXT-DAYNAME .", 
            "ngt7": "The TEAM-PLACE TEAM-NAME now have a couple days off , before they play host to the NEXT-OPPONENT-TEAM on NEXT-DAYNAME . ", 
            "ngt8": "The TEAM-NAME will look to bounce back when they take on NEXT-OPPONENT-TEAM-PLACE NEXT-OPPONENT-TEAM on NEXT-DAYNAME ."
        }

        # defeat templates
        self.dts = {
            "dt1": "The host HOME-TEAM-PLACE HOME-TEAM-NAME ( HOME-TEAM-WINS - HOME-TEAM-LOSSES ) defeated the visiting VIS-TEAM-PLACE VIS-TEAM-NAME ( VIS-TEAM-WINS - VIS-TEAM-LOSSES ) HOME-TEAM-PTS - VIS-TEAM-PTS at STADIUM on DAYNAME .",
            "dt2": "The visiting VIS-TEAM-PLACE VIS-TEAM-NAME ( VIS-TEAM-WINS - VIS-TEAM-LOSSES ) defeated the host HOME-TEAM-PLACE HOME-TEAM-NAME ( HOME-TEAM-WINS - HOME-TEAM-LOSSES ) VIS-TEAM-PTS - HOME-TEAM-PTS at STADIUM on DAYNAME .",
            "dt3": "VIS-LEADER_FIRST_NAME VIS-LEADER_SECOND_NAME paced the VIS-TEAM-PLACE VIS-TEAM-NAME to a VIS-TEAM-PTS - HOME-TEAM-PTS win against HOME-TEAM-PLACE HOME-TEAM-NAME on DAYNAME at STADIUM in HOME-TEAM-PLACE .",
            "dt4": "HOME-LEADER_FIRST_NAME HOME-LEADER_SECOND_NAME paced the HOME-TEAM-PLACE HOME-TEAM-NAME to a HOME-TEAM-PTS - VIS-TEAM-PTS win against VIS-TEAM-PLACE VIS-TEAM-NAME on DAYNAME at STADIUM in HOME-TEAM-PLACE ."
        }

    def generate_from_template(self, template, data):
        new_toks = []
        for tok in template.split(' '):
            if tok in data:
                new_toks.append(data[tok])
            else:
                new_toks.append(tok)
        return ' '.join(new_toks)        
    
    def get_defeat_template(self, data):
        # print(data)
        if data['HOME-WINNER'] == 'yes':
            if int(data['HOME-TEAM-PTS']) - int(data['VIS-TEAM-PTS']) > 10:
                template = self.dts[f"dt4"]
            else:
                template = self.dts[f"dt1"]
        else:
            if int(data['VIS-TEAM-PTS']) - int(data['HOME-TEAM-PTS']) > 10:
                template = self.dts[f"dt4"]
            else:
                template = self.dts[f"dt2"]

        return template

    def generate_defeat_sentence(self, item):
        data = {key: val for key, val in item['game'].items()}
        for key, val in item['home_line'].items():
            data[f'HOME-{key}'] = val
            data[f'VIS-{key}'] = item['vis_line'][key]
        template = self.get_defeat_template(data)
        return self.generate_from_template(template, data)

    def get_next_game_template(self, item, team='home'):
        line_score = item['home_line'] if team == 'home' else item['vis_line']
        next_game = item['home_next_game'] if team == 'home' else item['vis_next_game']

        data = line_score | next_game
        streak_count, team_win = line_score['STREAK-COUNT'], line_score['WINNER']
        this_game_date = datetime.strptime(f"{item['game']['YEAR']} {item['game']['MONTH']} {item['game']['DAY']}", '%Y %B %d').date()
        next_game_date = datetime.strptime(f"{next_game['NEXT-YEAR']} {next_game['NEXT-MONTH']} {next_game['NEXT-DAY']}", '%Y %B %d').date()
        days_for_next_game = (next_game_date - this_game_date).days

        if team == 'home' and next_game['NEXT-HOME-TEAM'] == item['home_name']: # C - H : N - H
            if team_win == 'yes' and streak_count > 2:
                temp = self.ngts["ngt2"]
            elif team_win == 'no':
                temp = self.ngts["ngt5"]
            else:
                temp = self.ngts["ngt1"]
        elif team == 'home' and next_game['NEXT-VISITING-TEAM'] == item['home_name']: # C - H : N - V
            temp = self.ngts["ngt4"]
                
        elif team == 'vis' and next_game['NEXT-HOME-TEAM'] == item['vis_name']: # C - V : N - H
            if team_win == 'yes' and streak_count > 2:
                temp = self.ngts["ngt2"]
            elif team_win == 'no':
                temp = self.ngts["ngt8"]
            else:
                temp = self.ngts["ngt6"]
        elif team == 'vis' and next_game['NEXT-VISITING-TEAM'] == item['vis_name']: # C - V : N - V
            if days_for_next_game > 3:
                temp = self.ngts["ngt7"]
            else:
                temp = self.ngts["ngt3"]

        return temp, data

    def generate_next_game_sentence(self, item):
        final_sent = ""

        if 'home_next_game' in item.keys():
            template, data = self.get_next_game_template(item, team='home')
            home_sent = self.generate_from_template(template, data)
            final_sent = home_sent

        if 'vis_next_game' in item.keys():
            template, data = self.get_next_game_template(item, team='vis')
            vis_sent = self.generate_from_template(template, data)
            final_sent = vis_sent

        if 'home_next_game' in item.keys() and 'vis_next_game' in item.keys():
            if vis_sent[0] == 'T':
                final_sent = f'{home_sent[:-1]} while t{vis_sent[1:]}'
            elif vis_sent[0] == 'N':
                final_sent = f'{vis_sent} {home_sent}'
            elif home_sent[0] == 'N':
                final_sent = f'{home_sent} while t{vis_sent}'

        return final_sent

# js = json.load(open('./data/jsons/2018_w_opp.json', 'r'))
# rfg = RulesForGeneration()
# for idx, i in enumerate(js):
#     if idx % 100 == 0:
#     # if idx == 300:
#         print(idx, rfg.generate_defeat_sentence(i))
#         # print(idx, rfg.generate_next_game_sentence(i))
#         print()
