from ext_team_temp import extract_team_templates_from_text
from player_stats_temp import extracting_player_stats_templates_from_texts
from team_stats_temp import extracting_team_stats_templates_from_texts

# print(f'extracting next game templates')
# extract_team_templates_from_text(cat='next-game')

# print(f'extracting team defeat templates')
# extract_team_templates_from_text(cat='defeat')

print(f'extracting players stats templates')
extracting_player_stats_templates_from_texts()

print(f'extracting teams stats templates')
extracting_team_stats_templates_from_texts()

print(f"all done")