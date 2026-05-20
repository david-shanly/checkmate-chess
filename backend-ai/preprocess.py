import json
import numpy as np
import chess
from tqdm import tqdm

INPUT_FILE = "data/lichess_1m_positions.jsonl"
OUTPUT_FILE = "dataset.npz"

MAX_POSITIONS = 1_000_000

piece_map = {
    'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
    'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
}

def board_to_tensor(board):

    tensor = np.zeros((13, 8, 8), dtype=np.float32)

    for square, piece in board.piece_map().items():

        row = 7 - (square // 8)
        col = square % 8

        tensor[piece_map[piece.symbol()], row, col] = 1

    # Side to move plane
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1

    return tensor


def main():

    X = []
    y = []

    count = 0

    with open(INPUT_FILE, "r") as f:

        for line in tqdm(f):

            if count >= MAX_POSITIONS:
                break

            try:
                data = json.loads(line)

                fen = data["fen"]
                eval_cp = data["eval"]

                board = chess.Board(fen)

                tensor = board_to_tensor(board)

                label = np.tanh(eval_cp / 400)

                X.append(tensor)
                y.append(label)

                count += 1

            except:
                continue

    X = np.array(X)
    y = np.array(y, dtype=np.float32)

    print("Saving dataset...")

    np.savez(OUTPUT_FILE, X=X, y=y)

    print("Done")
    print("X shape:", X.shape)
    print("y shape:", y.shape)


if __name__ == "__main__":
    main()