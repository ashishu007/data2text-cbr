"""
Select important players from the textual summaries.
"""

import json
import pickle
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

atts_js = json.load(open(f'./data/atts.json', 'r'))

class ImportantPlayers:
    def __init__(self):
        self.season_splits = {
            "train": [2014, 2015, 2016],
            "valid": [2017],
            "test": [2018]
        }
        self.num_players = 30
        self.num_ftrs = 29
        self.sim_ftrs = atts_js['box-score sim_ftrs keys']

    def vectorise_one_game_data(self, player_sim_ftrs):
        new_data = []
        for j in range(len(player_sim_ftrs)):
            tmp = []
            for z in range(j+1, self.num_players):
                for num in player_sim_ftrs[z]:
                    tmp.append(num)
            if j > 0:
                for z in range(0, j):
                    for num in player_sim_ftrs[z]:
                        tmp.append(num)
            new_data.append(np.insert(player_sim_ftrs[j], self.num_ftrs, tmp,axis=0))
        return np.array(new_data)

    def get_player_sim_ftrs_from_json_item(self, item):
        player_stats = []
        max_player_in_game = max_player_in_game = len(item['box_score']['FIRST_NAME'])

        for player_idx in range(self.num_players):
            useful_stats = []
            if player_idx < max_player_in_game:
                for k, v in item['box_score'].items():
                    if k in self.sim_ftrs:
                        if k not in ['STARTER', 'IS_HOME', 'DOUBLE_DOUBLE']:
                            val = int(v[str(player_idx)]) if v[str(player_idx)] != 'N/A' else 0
                            useful_stats.append(val)
                        else:
                            val = 1 if v == 'yes' else 0
                            useful_stats.append(val)
                if item['box_score']['IS_HOME'][str(player_idx)] == 'yes':
                    val = 1 if item['box_score']['PLAYER_NAME'][str(player_idx)] == f"{item['home_line']['LEADER_FIRST_NAME']} {item['home_line']['LEADER_SECOND_NAME']}" else 0
                else:
                    val = 1 if item['box_score']['PLAYER_NAME'][str(player_idx)] == f"{item['vis_line']['LEADER_FIRST_NAME']} {item['vis_line']['LEADER_SECOND_NAME']}" else 0
                useful_stats.append(val)
            
            else:
                useful_stats.extend([0.0] * len(self.sim_ftrs))
            player_stats.append(useful_stats)

        return self.vectorise_one_game_data(np.array(player_stats))

    def load_split_data(self, split='train'):
        seasons = self.season_splits[split]

        new_data, new_label = [], []
        for season in seasons:
            data = np.load(f'./data/imp_players/{season}_x_arr.npy')
            label = np.load(f'./data/imp_players/{season}_y_arr.npy')

            label1 = np.reshape(label, label.shape[0] * label.shape[1]).tolist()
            new_label.extend(label1)

            for i in range(len(data)):
                for j in range(len(data[i])):
                    tmp = []
                    for z in range(j+1, self.num_players):
                        for num in data[i][z]:
                            tmp.append(num)
                    if j > 0:
                        for z in range(0, j):
                            for num in data[i][z]:
                                tmp.append(num)
                    new_data.append(np.insert(data[i][j], self.num_ftrs, tmp,axis=0))
        
        x_arr = np.array(new_data)
        y_arr = np.array(new_label)

        return x_arr, y_arr
    
    def save_data_scaler_model(self):
        seasons = self.season_splits['train']
        all_data = [np.load(f'./data/imp_players/{season}_x_arr.npy') for season in seasons]
        all_data_array = np.concatenate(all_data)
        all_data_array = all_data_array.reshape(all_data_array.shape[0] * all_data_array.shape[1], all_data_array.shape[2])

        scaler_model = MinMaxScaler(feature_range=(0, 1))
        scaler_model.fit(all_data_array)
        # scaler.transform(data)

        scaler_filename = f"./data/imp_players/imp_player_data_scaler.pkl"
        with open(scaler_filename, 'wb') as file:
            pickle.dump(scaler_model, file)

    def select_imp_player_on_eff(self, box_score):
        """
        NBA's efficiency rating: (PTS + REB + AST + STL + BLK − ((FGA − FGM) + (FTA − FTM) + TO))
        """
        home_player_eff, vis_player_eff = {}, {}
        for k, v in box_score['PLAYER_NAME'].items():

            pts = int(box_score['PTS'][k]) if box_score['PTS'][k] != 'N/A' else 0
            reb = int(box_score['REB'][k]) if box_score['REB'][k] != 'N/A' else 0
            ast = int(box_score['AST'][k]) if box_score['AST'][k] != 'N/A' else 0
            stl = int(box_score['STL'][k]) if box_score['STL'][k] != 'N/A' else 0
            blk = int(box_score['BLK'][k]) if box_score['BLK'][k] != 'N/A' else 0
            fga = int(box_score['FGA'][k]) if box_score['FGA'][k] != 'N/A' else 0
            fgm = int(box_score['FGM'][k]) if box_score['FGM'][k] != 'N/A' else 0
            fta = int(box_score['FTA'][k]) if box_score['FTA'][k] != 'N/A' else 0
            ftm = int(box_score['FTM'][k]) if box_score['FTM'][k] != 'N/A' else 0
            to = int(box_score['TO'][k]) if box_score['TO'][k] != 'N/A' else 0

            eff = pts + reb + ast + stl + blk - ((fga - fgm) + (fta - ftm) + to)
            if box_score['IS_HOME'][k] == 'yes':
                home_player_eff[v] = eff
            else:
                vis_player_eff[v] = eff

        hpe = dict(sorted(home_player_eff.items(), key=lambda item: item[1], reverse=True))
        vpe = dict(sorted(vis_player_eff.items(), key=lambda item: item[1], reverse=True))        

        return hpe, vpe

    
    def train_model(self, model_name='lr'):
        train_x, train_y = self.load_split_data(split='train')
        test_x, test_y = self.load_split_data(split='valid')
        print(train_x.shape, train_y.shape)
        print(test_x.shape, test_y.shape)

        model = make_pipeline(
                        MinMaxScaler(feature_range=(0, 1)),
                        # StandardScaler(), 
                        LogisticRegression(
                            random_state=0,
                            class_weight='balanced', max_iter=500
                        )
                    )
        model.fit(train_x, train_y)

        clf_filename = f"./data/imp_players/imp_player_model_{model_name}.pkl"

        with open(clf_filename, 'wb') as file:
            pickle.dump(model, file)

        pred_y = model.predict(test_x)
        print('Valid Performance')
        print(f'Accuracy: {accuracy_score(test_y, pred_y)}\t\tMacroF1: {f1_score(test_y, pred_y, average="macro")}')
        # print(pred_y[:100], test_y[:100])

        print(model[1])
        # new_coefs = self.normalize_coef(model[1].coef_.reshape(-1, 1))
        new_coefs = model[1].coef_.reshape(-1, 1)
        for i in range(self.num_ftrs):
            for j in range(self.num_ftrs + i, self.num_ftrs * self.num_players, self.num_ftrs):
                new_coefs[i] = new_coefs[i] + new_coefs[j]
        new_coefs = new_coefs[:self.num_ftrs].tolist()
        print(new_coefs[:29])
        print(new_coefs[-1][0]/self.num_players)
        
        ftrs_weights = {ftr: new_coefs[idx][0]/self.num_players for idx, ftr in enumerate(self.sim_ftrs)}
        json.dump(ftrs_weights, open('./data/imp_players/ftr_weights.json', 'w'), indent='\t')

        return model

    def pred_sample(self, model, test_x):
        pred_y = model.predict(test_x)
        return pred_y

    def normalize_coef(self, coefs):
        coefsmin, coefsmax = min(coefs), max(coefs)
        for i, val in enumerate(coefs):
            coefs[i] = (val-coefsmin) / (coefsmax-coefsmin)
        return coefs

    def select_imp_player_by_model(self, clf, item):
        x_arr = self.get_player_sim_ftrs_from_json_item(item)
        
        # pkl_filename = f"./data/imp_players/imp_player_model_lr.pkl"
        # with open(pkl_filename, 'rb') as file:
        #     clf = pickle.load(file)
        
        pred_y = clf.predict(x_arr)
        
        home_ips, vis_ips = [], []
        for idx, i in enumerate(pred_y):
            if i == 1:
                if item['box_score']['IS_HOME'][str(idx)] == 'yes':
                    home_ips.append(item['box_score']['PLAYER_NAME'][str(idx)])
                else:
                    vis_ips.append(item['box_score']['PLAYER_NAME'][str(idx)])

        return home_ips, vis_ips


obj = ImportantPlayers()
# obj.save_data_scaler_model()
clf = obj.train_model()
# pkl_filename = f"./data/imp_players/imp_player_model_lr.pkl"
# with open(pkl_filename, 'rb') as file:
#     clf = pickle.load(file)
# tx, ty = obj.load_split_data('test')
# py = clf.predict(tx)
# print(f'Accuracy: {accuracy_score(ty, py)}\t\tMacroF1: {f1_score(ty, py, average="macro")}')
# print(ty[:100])
# print(py[:100])
# # for i, j in zip(range(0, len(tx) - obj.num_players, obj.num_players), range(obj.num_players, len(tx), obj.num_players)):
# #     print(i, j)
# #     print(ty[i:j])
# #     print(py[i:j])

# # print(py.shape, ty.shape)

