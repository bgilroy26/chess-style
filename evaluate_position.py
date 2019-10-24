import chess
import thai_life
import fen2vec4
import sys
import numpy as np
import torch

#bring in model
state = torch.load('last_evaluator.tar', map_location=torch.device('cpu'))
model = thai_life.ThaiLife()
model.load_state_dict(state['state_dict'])
model.eval()

#determine MaeToi scores for position
def get_MaeToi(fen):
    if ' ' in fen:
        fen = fen.split()[0]
    np_input_vector = fen2vec4.batchtotensor(fen)
    pytorch_input_vector = torch.from_numpy(np_input_vector).type(torch.FloatTensor)
    model_output = model(pytorch_input_vector).detach().numpy()[0]
    return (model_output[0], model_output[1], model_output[2])

if __name__ == '__main__':
    white_win, draw, black_win = get_MaeToi(sys.argv[1])
    print(white_win, draw, black_win)
    
