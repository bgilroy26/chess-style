import random
import pprint
import pptree
import chess
import sys
import collections
import evaluate_position

PlayerMove = collections.namedtuple('PlayerMove', ['position', 'score'])
OpponentMove = collections.namedtuple('OpponentMove', ['position', 'score'])
ChessLeaf = collections.namedtuple('ChessLeaf', ['position', 'tally'])

class ChessTree:
    def __init__(self, move_and_score):
        self.data = move_and_score
        self.children = []
    
    def add_branch(self, original_move_and_score, move_and_score):
        if original_move_and_score == self.data.position:
            self.children.append(ChessTree(move_and_score))
        else:
            for child in self.children:
                child.add_branch(original_move_and_score, move_and_score)

    def trim(self):
        if self.children == []:
            return
        if len(self.children) > 4:
            self.children = sorted(self.children, key=lambda x: x.data.score, reverse=True)[:4]
        else:
            for child in self.children:
                child.trim()

    def add_leaf(self, parent_position, leaf):
        if self.children == []:
            self.children = leaf
        elif isinstance(self.children, ChessLeaf):
            return
        else:
            for child in self.children:
                if child.data.position == parent_position:
                    child.add_leaf(parent_position, leaf)
                    
    def get_top_move(self):
        moves = []
        for move in self.children:
            moves.append((move.data.position, move.get_move_score()))
        final_move = sorted(moves, key=lambda x: x[1], reverse=True)[0] 
        return final_move[0]

    def get_move_score(self):
        positions = []
        negative = False
        if len(self.children) > 0:
            if isinstance(self.children[0].data, OpponentMove):
                negative = True
        for child in self.children:
            if isinstance(child, ChessLeaf):
                positions.append(child.position)
                break
        
        return sum([position.data.score for position in positions])
        
    def __str__(self):
        return f"{self.data.position} {self.data.score}"

class PrincipalVariation:
    def __init__(self, color, fen):
        self.board = chess.Board(fen=fen)
        self.color = color
        self.turn = color
        self.tree = ChessTree(OpponentMove(fen, 0))
    
    def update_fen(self, fen):
        self.board = chess.Board(fen=fen)

    def generate(self, depth):
        # an even numbered depth will always end on opponent's move
        # seed tree.children with first replies
        for idx, x in enumerate(range(depth)):
            if idx == depth-1:
                for move in self.board.legal_moves:
                    parent_position = self.board.fen()
                    self.board.push(move)
                    new_pos = self.board.fen().split()[0]
                    self.board.pop()
                    if self.color == 0:
                        white_win = evaluate_position.get_MaeToi(new_pos)[0]
                        self.tree.add_leaf(parent_position, ChessLeaf(new_pos, white_win))
                    else:
                        black_win = evaluate_position.get_MaeToi(new_pos)[2]
                        self.tree.add_leaf(parent_position, ChessLeaf(new_pos, black_win))

            else:
                parent_position = self.board.fen().split()[0]
                for move in self.board.legal_moves:
                    self.board.push(move)
                    new_pos = self.board.fen().split()[0]
                    self.board.pop()
                    if self.color == 0:
                        if self.turn == self.color:
                            white_win = evaluate_position.get_MaeToi(new_pos)[0]
                            self.tree.add_branch(parent_position, PlayerMove(new_pos, white_win))
                        else:
                            black_win = evaluate_position.get_MaeToi(new_pos)[2]
                            self.tree.add_branch(parent_position, OpponentMove(new_pos, black_win))
                    else:
                        if self.turn == self.color:
                            black_win = evaluate_position.get_MaeToi(new_pos)[2]
                            self.tree.add_branch(parent_position, PlayerMove(new_pos, black_win))
                        else:
                            white_win = evaluate_position.get_MaeToi(new_pos)[0]
                            self.tree.add_branch(parent_position, OpponentMove(new_pos, white_win))
                self.tree.trim() 
                self.turn += 1
                self.turn = self.turn % 2

        
    def evaluate(self):
        return self.tree.get_top_move()

def run(fen):
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
    
    pv = PrincipalVariation(0, fen)
    pv.generate(3)
    def print_chess_tree(tree):
        try:
            if tree.children:
                print(f"{tree.children} {print_chess_tree(tree.children)}")
        except AttributeError:
            print(tree)
    #return pv.evaluate()
    print_chess_tree(pv.tree)

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
        player_moves_and_scores.append(PlayerMove(new_pos, white_win))

    player_moves_and_scores_top_4 = sorted(player_moves_and_scores, key=lambda x: x.score, reverse=True)[:4]

    #add room for the opponent's replies
    player_moves_and_scores_top_4 = [[move_and_score, []] for move_and_score in player_moves_and_scores_top_4]
    for idx, first_move in enumerate(player_moves_and_scores_top_4):
        new_board = chess.Board(fen=first_move[0][0])
        for second_move in new_board.legal_moves:
            new_board.push(second_move)
            new_pos = new_board.fen().split()[0]
            new_board.pop()
            black_win = evaluate_position.get_MaeToi(new_pos)[2]
            player_moves_and_scores_top_4[idx][1].append(OpponentMove(new_pos, black_win))
    for idx in range(4):
        player_moves_and_scores_top_4[idx][1] = sorted(player_moves_and_scores_top_4[idx][1], key=lambda x: x.score, reverse=True)[:4]
    opponents_moves_and_scores_top_16 = player_moves_and_scores_top_4   

    for idx in range(4):
        for idy in range(4):
            opponents_moves_and_scores_top_16[idx][1][idy] = [opponents_moves_and_scores_top_16[idx][1][idy], []]

    for idx in range(4):
        for idy in range(4):
            new_board = chess.Board(fen=opponents_moves_and_scores_top_16[idx][1][idy][0].position)
            for third_move in new_board.legal_moves:
                new_board.push(third_move)
                new_pos = new_board.fen().split()[0]
                new_board.pop()
                white_win = evaluate_position.get_MaeToi(new_pos)[0]
                opponents_moves_and_scores_top_16[idx][1][idy][1].append(PlayerMove(new_pos, white_win))

    for idx in range(4):
        for idy in range(4):
            opponents_moves_and_scores_top_16[idx][1][idy][1] = sorted(opponents_moves_and_scores_top_16[idx][1][idy][1], key=lambda x: x.score, reverse=True)[0]

    player_moves_top_16 = opponents_moves_and_scores_top_16

    final_moves = []
    for idx in range(4):
        for idy in range(4):
            final_moves.append(PlayerMove(player_moves_top_16[idx][0].position, player_moves_top_16[idx][0].score - player_moves_top_16[idx][1][idy][0].score + player_moves_top_16[idx][1][idy][1].score))

    final_move = sorted(final_moves, key=lambda x:x[1], reverse=True)[0]
    return final_move.position

if __name__ == '__main__':
    fen = sys.argv[1]
    print(run(fen))
