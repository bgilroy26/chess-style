import chess
import sys

with open(sys.argv[1]) as f:
    lines = f.readlines()
player_first = sys.argv[2]
player_last = sys.argv[3]

player_index = 0
game_index = 0
position_index = 0
player_color_flag = 0
for idx, line in enumerate(lines[:200]):
    row = {}
    row['player_id'] = player_index
    if line == '\n':
        game_index += 1
        continue
    row['game_id'] = game_index
    row['position_id'] = position_index
    line_parts = line.split()
    row['psoition_fen'] = line_parts[0]
    if line_parts[-1] == '1-0':
        row['result_white_win'] = 1
        row['result_draw'] = 0
        row['result_black_win'] = 0
    elif line_parts[-1] == '1/2-1/2':
        row['result_white_win'] = 0
        row['result_draw'] = 1
        row['result_black_win'] = 0
    elif line_parts[-1] == '0-1':
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
        raise TypeError(f'game_result error line {idx}')

    player_active_squares = 0
    for i in range(64):
        if board.is_attacked_by(player_color_flag, i):
            player_active_squares += 1

    row['player_piece_activity'] = player_active_squares

    opp_active_squares = 0
    for i in range(64):
        if board.is_attacked_by(~player_color_flag, i):
            opp_active_squares += 1

    row['opponent_piece_activity'] = opp_active_squares
