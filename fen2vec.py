import numpy as np
import sys

def batchtotensor(inputbatch):
    
    pieces_str = "PNBRQK"
    pieces_str += pieces_str.lower()
    pieces = set(pieces_str)
    valid_spaces = set(range(1,9))
    pieces_dict = {pieces_str[0]:1, pieces_str[1]:2, pieces_str[2]:3, pieces_str[3]:4,
                    pieces_str[4]:5, pieces_str[5]:6,
                    pieces_str[6]:-1, pieces_str[7]:-2, pieces_str[8]:-3, pieces_str[9]:-4, 
                    pieces_str[10]:-5, pieces_str[11]:-6}

    maxnum = len(inputbatch)
    boardtensor = np.zeros((maxnum, 8, 8,7))
    
    for num, inputstr in enumerate(inputbatch):
        inputliste = inputstr.split()
        rownr = 0
        colnr = 0
        for i, c in enumerate(inputliste[0]):
            if c in pieces:
                boardtensor[num, rownr, colnr, np.abs(pieces_dict[c])-1] = np.sign(pieces_dict[c])
                colnr = colnr + 1
            elif c == '/':  # new row
                rownr = rownr + 1
                colnr = 0
            elif int(c) in valid_spaces:
                colnr = colnr + int(c)
            else:
                raise ValueError("invalid fenstr at index: {} char: {}".format(i, c))
        
        if inputliste[1] == "w":
            for i in range(8):
                for j in range(8):
                    boardtensor[num, i, j, 6] = 1
        else:
            for i in range(8):
                for j in range(8):
                    boardtensor[num, i, j, 6] = -1
  
    return boardtensor

if __name__ == '__main__':
    with open('../one_big_epd.epd', encoding='latin-1') as f:
        one_big_epd = f.readlines()
    fens = [line.split() for line in one_big_epd if line != '\n']
    data  = [(fen[0], fen[-1]) for fen in fens if fen[-1] != '*;']
    positions, results = zip(*data)
    final_results = []
    for result in results:
        result = result[:-1]
        if result == '1-0':
            final_results.append((1,0,0))
        if result == '1/2-1/2':
            final_results.append((0,1,0))
        if result == '0-1':
            final_results.append((0,0,1))

    to_save_positions = np.array(positions, dtype=str)
    to_save_results = np.array(final_results, dtype=int)
    #to_save = np.column_stack((to_save_positions, to_save_results))
    print(to_save_positions.shape, to_save_results.shape)
    
    np.save('all_chess_positions_just_pos', to_save_positions)
    np.save('all_chess_positions_just_results', to_save_results)
