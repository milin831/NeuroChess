import chess
import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import chess_ai 

app = FastAPI()

origins = [
    "*",  
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

POLICY_OUTPUT_SIZE = len(chess_ai.MOVE_TO_INDEX)
INPUT_CHANNELS = 14

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CHECKPOINT_PATH = 'chess_model.pth'
model = chess_ai.ChessModel(in_channels=INPUT_CHANNELS, policy_output_size=POLICY_OUTPUT_SIZE)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))

model.to(DEVICE)

class BoardState(BaseModel):
    fen: str

@app.post("/get_move")
async def get_ai_move(state: BoardState):
    if model is None:
        return {"error": "Model not loaded on the server."}
    try:
        board = chess.Board(state.fen)
        if board.is_game_over():
            return {"move": None, "status": "game_over"}
        
        ai_move = chess_ai.get_best_move(board, model, device=DEVICE)
        
        if ai_move:
            return {"move": ai_move.uci()}
        else:
            return {"move": None, "status": "no_move"}

    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def read_root():
    return {"message": "Chess AI API is running"}