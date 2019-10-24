import pprint
import chess
import sys
import evaluate_position

board = chess.Board(fen=sys.argv[1])

player_moves_and_scores = []
for idx, move in enumerate(board.legal_moves):
    board.push(move)
    new_pos = board.fen().split()[0]
    board.pop()
    white_win = evaluate_position.get_MaeToi(new_pos)[0]
    player_moves_and_scores.append((new_pos, white_win))

player_moves_and_scores_top_4 = sorted(player_moves_and_scores, key=lambda x: x[1], reverse=True)[:4]

#add room for the opponent's replies
player_moves_and_scores_top_4 = [[move_and_score, []] for move_and_score in player_moves_and_scores_top_4]
for idx, first_move in enumerate(player_moves_and_scores_top_4):
    new_board = chess.Board(fen=first_move[0][0])
    for second_move in new_board.legal_moves:
        new_board.push(second_move)
        new_pos = new_board.fen().split()[0]
        new_board.pop()
        black_win = evaluate_position.get_MaeToi(new_pos)[2]
        player_moves_and_scores_top_4[idx][1].append((new_pos, black_win))
for idx in range(4):
    player_moves_and_scores_top_4[idx][1] = sorted(player_moves_and_scores_top_4[idx][1], key=lambda x: x[1], reverse=True)[:4]
opponents_moves_and_scores_top_16 = player_moves_and_scores_top_4   

for idx in range(4):
    for idy in range(4):
        opponents_moves_and_scores_top_16[idx][1][idy] = [opponents_moves_and_scores_top_16[idx][1][idy], []]
pp = pprint.PrettyPrinter(indent=4)
pp.pprint(opponents_moves_and_scores_top_16)
sys.exit()
for idx, first_move in enumerate(opponents_moves_and_scores_top_16):
    for idy, second_move in enumerate(first_move[0]):
        if isinstance(second_move, tuple):
            new_board = chess.Board(fen=second_move[0])
            for third_move in new_board.legal_moves:
                new_board.push(third_move)
                new_pos = new_board.fen().split()[0]
                new_board.pop()
                white_win = evaluate_position.get_MaeToi(new_pos)[0]
    print(opponents_moves_and_scores_top_16[idx])
