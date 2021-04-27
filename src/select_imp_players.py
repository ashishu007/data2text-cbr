from imp_players_utility import ImportantPlayers
from misc import MiscTasks

mt = MiscTasks()
imp = ImportantPlayers()

# print(f'Save imp players data')
# mt.save_train_data_for_imp_players()

# print(f'Train the classifier')
# imp.tpot_best_imp_player_classifier()

print(f'save data scaler for teams')
mt.save_data_scaler_model(component='team')

print(f'save data scaler for players')
mt.save_data_scaler_model(component='player')

