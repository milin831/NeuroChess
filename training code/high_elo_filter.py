import zstandard as zstd
import chess.pgn
import io
import os
from tqdm import tqdm

input_path = 'lichess_db_standard_rated_2025-07.pgn.zst' 
output_path = 'high_elo.pgn'
ELO_THRESHOLD = 2400
CHUNK_SIZE = 1024 * 1024 * 4 

high_elo_cnt = 0
total_games_processed = 0

with open(output_path, 'w', encoding='utf-8') as output_file:
    with open(input_path, 'rb') as compressed_file:
        dctx = zstd.ZstdDecompressor()
        with dctx.stream_reader(compressed_file) as reader:
            buffer = ""
            is_first_chunk = True
            with tqdm(desc="Processing full games", unit=" games") as pbar:
                while True:
                    chunk = reader.read(CHUNK_SIZE)
                    if not chunk:
                        break 

                    buffer += chunk.decode('utf-8', errors='ignore')
                    
                    games_text = buffer.split('\n[Event "')
                    
                    buffer = games_text.pop()

                    for i, game_text_chunk in enumerate(games_text):
                        if not game_text_chunk.strip():
                            continue

                        if i == 0 and is_first_chunk:
                             full_game_text = game_text_chunk
                        else:
                             full_game_text = '[Event "' + game_text_chunk

                        pgn_io = io.StringIO(full_game_text)
                        
                        try:
                            headers = chess.pgn.read_headers(pgn_io)
                        except (ValueError, RuntimeError):
                            continue

                        if headers is None:
                            continue
                        
                        total_games_processed += 1
                        pbar.update(1)

                        white_elo = 0 if headers.get('WhiteElo', '?') == '?' else int(headers['WhiteElo'])
                        black_elo = 0 if headers.get('BlackElo', '?') == '?' else int(headers['BlackElo'])

                        avg_elo = 0
                        if white_elo != 0 and black_elo != 0:
                            avg_elo = (white_elo + black_elo) // 2
                        elif white_elo != 0:
                            avg_elo = white_elo
                        else:
                            avg_elo = black_elo

                        if avg_elo >= ELO_THRESHOLD:
                            high_elo_cnt += 1
                            print(full_game_text, file=output_file, end='\n\n')
                            pbar.set_postfix(found=f"{high_elo_cnt:,}")

                    is_first_chunk = False # After the first chunk, this is always false
                            
print("\n--- Filtering Complete ---")
print(f"Total valid games processed: {total_games_processed}")
print(f"High-ELO games found and saved: {high_elo_cnt}")