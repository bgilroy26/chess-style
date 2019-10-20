import chess
import sys

board = chess.Board(fen=sys.argv[1])
white_running_total = 0
for i in chess.scan_forward(board.occupied_co[True]):
    white_running_total +=  sum(list(map(int, bin(board.attackers_mask(True, i))[2:])))

black_running_total = 0
for i in chess.scan_forward(board.occupied_co[False]):
    black_running_total +=  sum(list(map(int, bin(board.attackers_mask(False, i))[2:])))

print(white_running_total, black_running_total)
