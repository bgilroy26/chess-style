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
    boardtensor = np.zeros((maxnum, 8, 8, 6))
    
    for num, inputstr in enumerate(inputbatch):
        rownr = 0
        colnr = 0
        for i, c in enumerate(inputstr):
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
        
    return boardtensor

if __name__ == '__main__':
    positions = np.load('all_chess_positions_just_pos.npy')
    positions = positions.tolist()
    to_save = batchtotensor(positions)
    len_to_save = len(to_save)
    num_to_train = int(len_to_save*.8)
    train = np.array(to_save[:num_to_train])
    test = np.array(to_save[num_to_train:])
    np.save('train', train)
    np.save('test', test)

