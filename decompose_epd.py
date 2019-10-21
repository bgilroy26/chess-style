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
