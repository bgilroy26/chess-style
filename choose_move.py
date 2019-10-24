import random
import pprint
import chess
import sys
import evaluate_position

def execute(fen):
    #hardcode openings
    openings = ['rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR',\
            'rnbqkbnr/pppppppp/8/8/2P5/8/PP1PPPPP/RNBQKBNR',\
            'rnbqkbnr/pppppppp/8/8/8/5N2/PPPPPPPP/RNBQKB1R',\
            'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR']
    if ' ' in fen: 
        if fen.split()[0] == 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR':
            return openings[random.randint(0,3)]
    else:
        if fen == 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR':
            return openings[random.randint(0,3)]

           
    board = chess.Board(fen=fen)

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

    for idx in range(4):
        for idy in range(4):
            new_board = chess.Board(fen=opponents_moves_and_scores_top_16[idx][1][idy][0][0])
            for third_move in new_board.legal_moves:
                new_board.push(third_move)
                new_pos = new_board.fen().split()[0]
                new_board.pop()
                white_win = evaluate_position.get_MaeToi(new_pos)[0]
                opponents_moves_and_scores_top_16[idx][1][idy][1].append((new_pos, white_win))

    for idx in range(4):
        for idy in range(4):
            opponents_moves_and_scores_top_16[idx][1][idy][1] = sorted(opponents_moves_and_scores_top_16[idx][1][idy][1], key=lambda x: x[1], reverse=True)[0]

    player_moves_top_16 = opponents_moves_and_scores_top_16

    final_moves = []
    for idx in range(4):
        for idy in range(4):
            final_moves.append((player_moves_top_16[idx][0][0], player_moves_top_16[idx][0][1] - player_moves_top_16[idx][1][idy][0][1] + player_moves_top_16[idx][1][idy][1][1]))

    final_move = sorted(final_moves, key=lambda x:x[1], reverse=True)[0]
    return final_move[0]
