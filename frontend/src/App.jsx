import { useState, useRef } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";

function ResultDisplay({ game, setChessPosition }) {
  let result = ""; 

  if (game.isGameOver()) {
    if (game.isCheckmate()) {
      result = game.turn() === "w" ? "Black Won!" : "White Won!";
    } else if (game.isStalemate()) {
      result = "Stalemate! Game Over.";
    } else if (game.isDraw()) {
      result = "Draw! Game Over.";
    }
  }

  const handleNewGame = () => {
    game.reset();
    setChessPosition(game.fen());
  };

  return (
    <div className="overlay">
      <div className="result-panel">
        <h3>{result}</h3>
        <button onClick={handleNewGame}>New Game</button>
      </div>
    </div>
  );
}

function App() {
  const chessGameRef = useRef(new Chess());
  const [chessPosition, setChessPosition] = useState(
    chessGameRef.current.fen()
  );

  const getMoveFromServer = async (currentFen) => {
    try {
      const response = await fetch("http://localhost:8000/get_move", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          fen: currentFen,
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.move) {
        const game = chessGameRef.current;
        game.move(data.move);
        setChessPosition(game.fen());
      } else if (data.status == "game_over") {
      }
    } catch (error) {
      console.error("Error fetching data:", error);
    }
  };

  function onPieceDrop({ sourceSquare, targetSquare }) {
    if (!targetSquare) {
      return false;
    }

    try {
      const game = chessGameRef.current;
      const move = game.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: "q",
      });

      if (move === null) return false;

      const newFen = game.fen();
      setChessPosition(newFen);

      setTimeout(() => getMoveFromServer(newFen), 300);

      return true;
    } catch (error) {
      console.log(error);
      return false;
    }
  }

  const chessboardOptions = {
    position: chessPosition,
    onPieceDrop,
    id: "play-vs-random",
    boardStyle:{width: 400},
    darkSquareStyle:{ backgroundColor: '#00c8ffff' },
    lightSquareStyle:{ backgroundColor: '#ffffffff' }
  };
  return (
    <div
      style={{
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: '100vh', 
      width: '100vw',    
      padding: '10px'    
    }}
    >
      {chessGameRef.current.isGameOver() && (
        <ResultDisplay
          game={chessGameRef.current}
          setChessPosition={setChessPosition}
        />
      )}
      <h1>Neural Chess: Your Move, Predicted.</h1>
      <Chessboard options={chessboardOptions} />;
      <button
        onClick={() => getMoveFromServer(chessPosition)}
        style={{ marginTop: "1rem", padding: "10px 20px", fontSize: "1rem" }}
      >
        Get AI Move for Current Position
      </button>
    </div>
  );
}

export default App;
