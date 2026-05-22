import chess
import chess.pgn
import numpy as np
import h5py
from collections import deque
from tqdm import tqdm

# --- Configuration ---
PGN_FILE_PATH = 'high_elo.pgn'
OUTPUT_HDF5_PATH = "chess_dataset.h5"
BATCH_SIZE = 2048
NUM_HISTORY_STATES = 0

# --- Helper Functions ---

def create_move_map():
    # Your function is kept as is, but the '' for promotion is non-standard.
    # It works because python-chess's move.uci() on a promotion includes the piece.
    move_to_index = {}
    index = 0
    for from_sq in chess.SQUARES:
        for to_sq in chess.SQUARES:
            if from_sq == to_sq:
                continue
            is_promotion = (
                (chess.square_rank(from_sq) == 6 and chess.square_rank(to_sq) == 7 and abs(chess.square_file(from_sq) - chess.square_file(to_sq)) <= 1) or
                (chess.square_rank(from_sq) == 1 and chess.square_rank(to_sq) == 0 and abs(chess.square_file(from_sq) - chess.square_file(to_sq)) <= 1)
            )
            base_uci = chess.square_name(from_sq) + chess.square_name(to_sq)
            if is_promotion:
                for piece in ['', 'q', 'r', 'b', 'n']:
                    move_to_index[base_uci + piece] = index
                    index += 1
            else:
                move_to_index[base_uci] = index
                index += 1
    return move_to_index

MOVE_TO_INDEX = create_move_map()

def bitboard_to_numpy(bb: int) -> np.ndarray:
    return np.unpackbits(np.array([bb], dtype=np.uint64).view(np.uint8), bitorder='little').reshape(8, 8)

def chess_board_to_numpy(board: chess.Board) -> np.ndarray:
    planes = np.zeros((12, 8, 8), dtype=np.uint8)
    for piece_type in chess.PIECE_TYPES:
        for color in chess.COLORS:
            bb = board.pieces_mask(piece_type, color)
            if bb:
                plane_idx = (piece_type - 1) * 2 + (1 - color)
                planes[plane_idx] = bitboard_to_numpy(bb)
    return planes

def move_to_label(move: chess.Move) -> int:
    return MOVE_TO_INDEX[move.uci()]

# --- Main Processing Workflow ---

# Correctly calculated feature shape
# 12 planes * N history states + 12 planes for current state + 1 turn plane + 1 castling plane
FEATURE_CHANNELS = 14
FEATURE_SHAPE = (FEATURE_CHANNELS, 8, 8)

batch_features = []
batch_labels = []

with h5py.File(OUTPUT_HDF5_PATH, 'w') as hf:
    dset_features = hf.create_dataset('features', (0,) + FEATURE_SHAPE, maxshape=(None,) + FEATURE_SHAPE, compression="gzip", dtype=np.uint8, chunks=True)
    dset_labels = hf.create_dataset('labels', (0,), maxshape=(None,), compression="gzip", dtype=np.int32, chunks=True)

    with open(PGN_FILE_PATH) as pgn:
        with tqdm(desc="Processing Games", unit=" games") as pbar:
            while True:
                game = chess.pgn.read_game(pgn)
                if not game:
                    break

                board = game.board()

                for move in game.mainline_moves():

                    current_planes = chess_board_to_numpy(board)
                    turn_plane = np.ones((1, 8, 8), dtype=np.uint8) if board.turn == chess.WHITE else np.zeros((1, 8, 8), dtype=np.uint8)
                    castling_plane = np.expand_dims(bitboard_to_numpy(board.castling_rights), axis=0).astype(np.uint8)
                    
                    feature = np.vstack([current_planes, turn_plane, castling_plane])
                    label = move_to_label(move)
                    
                    batch_features.append(feature)
                    batch_labels.append(label)

                    board.push(move)
                    

                if len(batch_features) >= BATCH_SIZE:
                    dset_features.resize(dset_features.shape[0] + len(batch_features), axis=0)
                    dset_features[-len(batch_features):] = batch_features
                    
                    dset_labels.resize(dset_labels.shape[0] + len(batch_labels), axis=0)
                    dset_labels[-len(batch_labels):] = batch_labels
                    
                    batch_features.clear()
                    batch_labels.clear()

                pbar.update(1)

            # 4. Save any remaining data from the last batch
            if batch_features:
                dset_features.resize(dset_features.shape[0] + len(batch_features), axis=0)
                dset_features[-len(batch_features):] = batch_features
                dset_labels.resize(dset_labels.shape[0] + len(batch_labels), axis=0)
                dset_labels[-len(batch_labels):] = batch_labels

print(f"Dataset creation complete. Saved to {OUTPUT_HDF5_PATH}")