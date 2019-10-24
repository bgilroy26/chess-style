from flask import request
from flask import Flask 
from flask_cors import CORS
import choose_move

app = Flask(__name__)
CORS(app)

@app.route('/')
def get_move():
    fen = request.args.get('fen')
    return choose_move.execute(fen)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8000', debug=True)
