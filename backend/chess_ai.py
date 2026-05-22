# --- Helper Functions ---
import chess
import h5py
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

def create_move_map():
    move_to_index = {}
    index_to_move = {}
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
                    index_to_move[index] = base_uci + piece
                    index += 1
            else:
                move_to_index[base_uci] = index
                index_to_move[index] = base_uci
                index += 1
    return move_to_index, index_to_move

MOVE_TO_INDEX, INDEX_TO_MOVE = create_move_map()

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


class ChessDataset(Dataset):
    def __init__(self, hdf5_path):
        self.h5_file = h5py.File(hdf5_path, 'r')
        self.features = self.h5_file['features']
        self.labels = self.h5_file['labels']

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature_data = self.features[idx]
        label_data = self.labels[idx]
        
        feature_tensor = torch.from_numpy(feature_data.astype(np.float32))
        label_tensor = torch.tensor(label_data, dtype=torch.long)
        
        return feature_tensor, label_tensor

class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class ChessModel(nn.Module):
    def __init__(self, in_channels, policy_output_size):
        super(ChessModel, self).__init__()
        num_filters = 256
        num_residual_blocks = 19
        
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        self.residual_tower = nn.Sequential(*[ResidualBlock(num_filters) for _ in range(num_residual_blocks)])
        self.policy_head = nn.Linear(num_filters * 8 * 8, policy_output_size)

    def forward(self, x):
        out = self.initial_conv(x)
        out = self.residual_tower(out)
        out = out.view(out.size(0), -1) 
        policy_logits = self.policy_head(out)
        return policy_logits
    
def get_best_move(board, model, device='cuda'):
    model.eval() # Set the model to evaluation mode

    current_planes = chess_board_to_numpy(board)
    turn_plane = np.ones((1, 8, 8), dtype=np.uint8) if board.turn == chess.WHITE else np.zeros((1, 8, 8), dtype=np.uint8)
    castling_plane = np.expand_dims(bitboard_to_numpy(board.castling_rights), axis=0).astype(np.uint8)
    
    feature_tensor = np.vstack([current_planes, turn_plane, castling_plane])
    
    input_tensor = torch.from_numpy(feature_tensor.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
    
    probabilities = F.softmax(logits, dim=1).cpu().numpy().flatten()
    
    legal_moves_mask = np.zeros_like(probabilities)
    for move in board.legal_moves:
        legal_moves_mask[MOVE_TO_INDEX[move.uci()]] = 1
    
    masked_probabilities = probabilities * legal_moves_mask

    if np.sum(masked_probabilities) > 0:
        masked_probabilities /= np.sum(masked_probabilities) # Re-normalize
        best_move_index = np.argmax(masked_probabilities)
        best_move_uci = INDEX_TO_MOVE[best_move_index]
        return chess.Move.from_uci(best_move_uci)
    else:
        return list(board.legal_moves)[0]