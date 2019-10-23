import chess
import generate_protectors
import thai_life
import fen2vec3
import sys
import numpy as np
import pandas as pd
import torch

#bring in model
state = torch.load('last_evaluator.tar', map_location=torch.device('cpu'))
model = thai_life.ThaiLife()
model.load_state_dict(state['state_dict'])
model.eval()

with open(sys.argv[1]) as f:
    lines = f.readlines()
player_first = sys.argv[2]
player_last = sys.argv[3]
player_index = int(sys.argv[4])

player_positions = []

game_index = 0
position_index = 0
player_color_flag = 0
check_name_flag = 1

sys.stdout.write("[")
sys.stdout.flush()
for idx, line in enumerate(lines):
    #there is a newline between every game
    if line == '\n':
        game_index += 1
        continue

    if idx % 25 == 0:
        sys.stdout.write("#")
        sys.stdout.flush()
    #since I am the one typing in names, I have to check my spelling
    if check_name_flag == 1:
        if player_first not in line:
            raise ValueError
        if player_last not in line:
            raise ValueError
        check_name_flag = 0
    

    #initialize row + metadata
    row = {}
    row['player_id'] = player_index
    row['game_id'] = game_index
    row['position_id'] = position_index

    #save position
    line_parts = line.split()
    row['position_fen'] = line_parts[0]
    row['whose_turn'] = line_parts[1]

    #get result of game
    if line_parts[-1] == '1-0;':
        row['result_white_win'] = 1
        row['result_draw'] = 0
        row['result_black_win'] = 0
    elif line_parts[-1] == '1/2-1/2;':
        row['result_white_win'] = 0
        row['result_draw'] = 1
        row['result_black_win'] = 0
    elif line_parts[-1] == '0-1;':
        row['result_white_win'] = 0
        row['result_draw'] = 0
        row['result_black_win'] = 1
    else:
        row['result_white_win'] = 0
        row['result_draw'] = 0
        row['result_black_win'] = 0

    #generate board
    board = chess.Board(fen=line_parts[0])
        
    #figure out which side of the board
    #player of interest is playing on
    if player_last + '-' in line:
        player_color_flag = True
    elif '-' + player_first in line:
        player_color_flag = False
    else:
        sys.stdout.write('\n')
        sys.stdout.write(str(idx) + ' --- ' + line)
        sys.stdout.write('\n')
        sys.stdout.flush()

    if player_color_flag:
        row['player_color'] = 'white'
    else:
        row['player_color'] = 'black'

    #determine piece activity for player
    player_active_squares = 0
    for i in range(64):
        if board.is_attacked_by(player_color_flag, i):
            player_active_squares += 1

    row['player_piece_activity'] = player_active_squares

    #determine piece activity for opponent
    opp_active_squares = 0
    for i in range(64):
        if board.is_attacked_by(~player_color_flag, i):
            opp_active_squares += 1

    row['opponent_piece_activity'] = opp_active_squares

    #determine piece coordination for player and opponent
    protection = generate_protectors.execute(row['position_fen'])

    row['player_protected_pieces'], row['opponent_protected_pieces'] = \
        protection[player_color_flag], protection[~player_color_flag]
    

    #determine MaeToi scores for position
    np_input_vector = fen2vec3.batchtotensor([row['position_fen']])
    pytorch_input_vector = torch.from_numpy(np_input_vector).type(torch.FloatTensor)
    model_output = model(pytorch_input_vector).detach().numpy()
    row['MaeToi_white_win'] = model_output[0][0]
    row['MaeToi_black_win'] = model_output[0][2]
    row['MaeToi_draw'] = model_output[0][1]
    
    player_positions.append(row)
    position_index += 1

#write results to csv
pos_df = pd.DataFrame(player_positions)
pos_df.to_csv('botvinnik_df.csv', index=False)
