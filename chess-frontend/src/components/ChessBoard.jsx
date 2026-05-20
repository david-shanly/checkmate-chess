// src/components/ChessBoard.jsx
import React, { useState, useEffect, useRef } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import * as ort from "onnxruntime-web";

// Mapping for ONNX tensor planes (12 planes for pieces, 1 for side to move)
const pieceMap = {
  'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
  'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
};

/**
 * Converts a chess.js game state to a (13, 8, 8) flat Float32Array for ONNX.
 */
function boardToTensor(chessInstance) {
  const tensor = new Float32Array(13 * 8 * 8);
  const board = chessInstance.board(); // 8x8 representation

  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const piece = board[r][c];
      if (piece) {
        // Uppercase for White pieces, lowercase for Black
        const symbol = piece.color === 'w' ? piece.type.toUpperCase() : piece.type.toLowerCase();
        const plane = pieceMap[symbol];
        if (plane !== undefined) {
          const idx = plane * 64 + r * 8 + c;
          tensor[idx] = 1.0;
        }
      }
    }
  }

  // Turn plane (plane 12) – all 1s if White to move, all 0s if Black
  if (chessInstance.turn() === 'w') {
    for (let i = 0; i < 64; i++) {
      tensor[12 * 64 + i] = 1.0;
    }
  }

  return tensor;
}

/**
 * Finds the square of the king of the current turn's active side.
 */
function getKingSquare(chessInstance) {
  const turnColor = chessInstance.turn();
  const board = chessInstance.board();
  for (let r = 0; r < 8; r++) {
    for (let c = 0; c < 8; c++) {
      const piece = board[r][c];
      if (piece && piece.type === "k" && piece.color === turnColor) {
        const files = ["a", "b", "c", "d", "e", "f", "g", "h"];
        const ranks = ["8", "7", "6", "5", "4", "3", "2", "1"];
        return `${files[c]}${ranks[r]}`;
      }
    }
  }
  return null;
}

export default function ChessBoard({ onMove, onAnalysisUpdate, difficulty, gameStarted, resetTrigger, playerColor = "w", boardThemeName = "slate" }) {
  const [game, setGame] = useState(() => new Chess());
  const gameRef = useRef(game);

  // Keep gameRef synchronized with game state to prevent stale closures
  useEffect(() => {
    gameRef.current = game;
  }, [game]);
  const [session, setSession] = useState(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [isAiThinking, setIsAiThinking] = useState(false);
  const thinkingTimeoutRef = useRef(null);
  const bestPossibleValueRef = useRef(0.0);

  // States for player pawn promotion
  const [showPromotionDialog, setShowPromotionDialog] = useState(false);
  const [promotionSquare, setPromotionSquare] = useState(null);
  const [moveFrom, setMoveFrom] = useState(null);
  const [moveTo, setMoveTo] = useState(null);
  const moveFromRef = useRef(null);
  const moveToRef = useRef(null);

  const fen = game.fen();

  // Helper to run deep position evaluation for Value and Policy
  const analyzePosition = async (currentChess, turnColor, sess) => {
    if (!sess) return { evaluation: 0.0, topMoves: [], bestValue: 0.0 };

    const legalMoves = currentChess.moves({ verbose: true });
    if (legalMoves.length === 0) {
      const isDraw = currentChess.isDraw();
      const val = isDraw ? 0.0 : currentChess.turn() === 'b' ? 1.0 : -1.0; // checkmate values
      return { evaluation: val, topMoves: [], bestValue: val };
    }

    try {
      // 1. Evaluate current position
      const currentTensor = boardToTensor(currentChess);
      const curInput = new ort.Tensor("float32", currentTensor, [1, 13, 8, 8]);
      const curRes = await sess.run({ input: curInput });
      const currentEval = curRes.value.data[0];

      // 2. Evaluate all legal moves in search of the best candidate
      const evaluatedMoves = [];
      for (const move of legalMoves) {
        const tempGame = new Chess(currentChess.fen());
        tempGame.move({ from: move.from, to: move.to, promotion: move.promotion || "q" });

        const tensorData = boardToTensor(tempGame);
        const inputTensor = new ort.Tensor("float32", tensorData, [1, 13, 8, 8]);
        const results = await sess.run({ input: inputTensor });
        const val = results.value.data[0];

        evaluatedMoves.push({
          san: move.san,
          val: val,
          from: move.from,
          to: move.to,
          promotion: move.promotion || "q"
        });
      }

      // Sort moves: if White's turn, maximize; if Black's turn, minimize
      if (turnColor === 'w') {
        evaluatedMoves.sort((a, b) => b.val - a.val);
      } else {
        evaluatedMoves.sort((a, b) => a.val - b.val);
      }

      const bestValue = evaluatedMoves[0].val;

      // Softmax scaling to simulate Policy Head probabilities (s = 4.0 temp)
      const scale = turnColor === 'w' ? 4.0 : -4.0;
      let expSum = 0;
      const movesWithExp = evaluatedMoves.map((m) => {
        const exp = Math.exp(scale * m.val);
        expSum += exp;
        return { ...m, exp };
      });

      const topCandidates = movesWithExp.map((m) => ({
        san: m.san,
        val: m.val,
        prob: expSum > 0 ? m.exp / expSum : 1 / legalMoves.length
      })).slice(0, 3);

      return {
        evaluation: currentEval,
        topMoves: topCandidates,
        bestValue: bestValue
      };
    } catch (err) {
      console.error("❌ Position analysis error:", err);
      return { evaluation: 0.0, topMoves: [], bestValue: 0.0 };
    }
  };

  // 1️⃣ Load ONNX model on mount
  useEffect(() => {
    async function loadModel() {
      try {
        setIsModelLoading(true);
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.1/dist/";

        const sess = await ort.InferenceSession.create("/chess_model.onnx");
        setSession(sess);
        setIsModelLoading(false);
        console.log("🤖 ONNX Chess Engine initialized.");
      } catch (err) {
        console.error("❌ Failed to load ONNX model:", err);
        setIsModelLoading(false);
      }
    }
    loadModel();

    return () => {
      if (thinkingTimeoutRef.current) clearTimeout(thinkingTimeoutRef.current);
    };
  }, []);

  // 2️⃣ Handle board resets from Parent
  useEffect(() => {
    setGame(new Chess());
    setIsAiThinking(false);
    bestPossibleValueRef.current = 0.0;
    setShowPromotionDialog(false);
    setPromotionSquare(null);
    setMoveFrom(null);
    setMoveTo(null);
    moveFromRef.current = null;
    moveToRef.current = null;
    if (thinkingTimeoutRef.current) clearTimeout(thinkingTimeoutRef.current);
  }, [resetTrigger]);

  // 3️⃣ Check for Checkmate/Stalemate/Draw
  const checkGameStatus = (currentGame) => {
    if (currentGame.isCheckmate()) {
      const winner = currentGame.turn() === 'b'
        ? (playerColor === 'w' ? "White (You)" : "White (Neural Engine)")
        : (playerColor === 'w' ? "Black (Neural Engine)" : "Black (You)");
      return { isOver: true, message: `Checkmate! ${winner} wins!` };
    }
    if (currentGame.isDraw()) {
      let reason = "Draw (Tie)";
      if (currentGame.isStalemate()) reason = "Draw (Stalemate)";
      else if (currentGame.isInsufficientMaterial()) reason = "Draw (Insufficient material)";
      else if (currentGame.isThreefoldRepetition()) reason = "Draw (Threefold repetition)";
      return { isOver: true, message: reason };
    }
    return { isOver: false, message: "" };
  };

  // 4️⃣ AI move trigger on active AI turn
  useEffect(() => {
    if (!gameStarted || isModelLoading) return;

    const aiColor = playerColor === 'w' ? 'b' : 'w';
    if (game.turn() === aiColor && !game.isGameOver()) {
      setIsAiThinking(true);

      thinkingTimeoutRef.current = setTimeout(async () => {
        await makeAiMove();
      }, 700);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fen, gameStarted, isModelLoading, playerColor]);

  // 5️⃣ Real-time background analysis on Player's active turn
  useEffect(() => {
    if (!gameStarted || isModelLoading || !session) return;

    if (game.turn() === playerColor && !game.isGameOver()) {
      const runPlayerAnalysis = async () => {
        const analysis = await analyzePosition(game, playerColor, session);
        bestPossibleValueRef.current = analysis.bestValue;

        if (onAnalysisUpdate) {
          onAnalysisUpdate(analysis);
        }
      };
      runPlayerAnalysis();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [fen, gameStarted, isModelLoading, session, playerColor]);

  // 6️⃣ Perform AI Move Calculation
  const makeAiMove = async () => {
    const legalMoves = game.moves({ verbose: true });
    if (legalMoves.length === 0) {
      setIsAiThinking(false);
      return;
    }

    // Evaluate position to find the best move & update side panel
    const aiColor = playerColor === 'w' ? 'b' : 'w';
    const analysis = await analyzePosition(game, aiColor, session);
    if (onAnalysisUpdate) {
      onAnalysisUpdate(analysis);
    }

    let chosenMove = null;

    const elo = Number(difficulty);
    let randomChance = 0.0;
    if (elo <= 1000) randomChance = 0.55;
    else if (elo <= 1200) randomChance = 0.40;
    else if (elo <= 1500) randomChance = 0.25;
    else if (elo <= 1800) randomChance = 0.12;
    else if (elo <= 2100) randomChance = 0.05;
    else randomChance = 0.00;

    const isRandomMove = Math.random() < randomChance;

    if (isRandomMove || analysis.topMoves.length === 0) {
      const randomIndex = Math.floor(Math.random() * legalMoves.length);
      chosenMove = legalMoves[randomIndex];
      console.log("🤖 AI: Selected random move (easy rating / fallback)");
    } else {
      const bestMoveSan = analysis.topMoves[0].san;
      chosenMove = legalMoves.find((m) => m.san === bestMoveSan) || legalMoves[0];
      console.log(`🤖 AI: Decided best move = ${chosenMove.san}`);
    }

    if (chosenMove) {
      try {
        const moveResult = game.move({
          from: chosenMove.from,
          to: chosenMove.to,
          promotion: chosenMove.promotion || "q",
        });

        if (moveResult) {
          const nextGame = new Chess(game.fen());
          setGame(nextGame);

          const status = checkGameStatus(nextGame);
          setIsAiThinking(false);

          if (onMove) {
            onMove(moveResult.san, playerColor, status.isOver, status.message);
          }
        }
      } catch (e) {
        console.error("AI move error:", e);
        setIsAiThinking(false);
      }
    } else {
      setIsAiThinking(false);
    }
  };

  // 7️⃣ Handle player's piece drops
  const handleDrop = async (sourceSquare, targetSquare) => {
    const activeGame = gameRef.current;
    if (!gameStarted || isAiThinking || isModelLoading || activeGame.turn() !== 'w' || activeGame.isGameOver()) {
      return false;
    }

    // Intercept pawn promotion moves to display the piece selection dialog
    const isPawn = activeGame.get(sourceSquare)?.type === 'p';
    const isPromotionRank = targetSquare[1] === '8';

    // Verify legality using a temporary chess state
    const tempGame = new Chess(activeGame.fen());
    let legalMove = null;
    try {
      legalMove = tempGame.move({ from: sourceSquare, to: targetSquare, promotion: 'q' });
    } catch (e) { }

    if (legalMove && isPawn && isPromotionRank) {
      setMoveFrom(sourceSquare);
      setMoveTo(targetSquare);
      moveFromRef.current = sourceSquare;
      moveToRef.current = targetSquare;
      setPromotionSquare(targetSquare);
      setShowPromotionDialog(true);
      return false; // Stop finalization to let user choose
    }

    try {
      const move = activeGame.move({
        from: sourceSquare,
        to: targetSquare,
        promotion: "q", // default fallback
      });

      if (move === null) return false;

      const nextGame = new Chess(activeGame.fen());
      setGame(nextGame);

      const status = checkGameStatus(nextGame);

      // Evaluate precision loss & move classification
      let loss = 0.0;
      let classification = "Excellent";

      if (session) {
        try {
          const nextTensor = boardToTensor(nextGame);
          const inputTensor = new ort.Tensor("float32", nextTensor, [1, 13, 8, 8]);
          const results = await session.run({ input: inputTensor });
          const playedValue = results.value.data[0];

          const bestVal = bestPossibleValueRef.current ?? playedValue;
          loss = Math.max(0, bestVal - playedValue);

          if (loss <= 0.05) classification = "Excellent";
          else if (loss <= 0.15) classification = "Good";
          else if (loss <= 0.35) classification = "Inaccuracy";
          else if (loss <= 0.65) classification = "Mistake";
          else classification = "Blunder";

          console.log(`🎯 Move Played: ${move.san} | Eval: ${playedValue.toFixed(2)} | Best possible: ${bestVal.toFixed(2)} | Loss: ${loss.toFixed(4)} | Class: ${classification}`);
        } catch (e) {
          console.error("Error evaluating move drop classification:", e);
        }
      }

      if (onMove) {
        onMove(move.san, playerColor === 'w' ? 'b' : 'w', status.isOver, status.message, loss, classification);
      }
      return true;
    } catch (err) {
      console.warn("Attempted illegal move:", err.message);
      return false;
    }
  };

  const handlePromotionPieceSelect = (piece, promoteFromSquare, promoteToSquare) => {
    if (!piece) {
      setShowPromotionDialog(false);
      setPromotionSquare(null);
      return false;
    }

    const activeGame = gameRef.current;
    const fromSquare = promoteFromSquare || moveFromRef.current || moveFrom;
    const toSquare = promoteToSquare || moveToRef.current || moveTo;

    if (!fromSquare || !toSquare) {
      console.warn("Attempted promotion without valid from/to squares:", { fromSquare, toSquare });
      setShowPromotionDialog(false);
      setPromotionSquare(null);
      return false;
    }

    // Map react-chessboard promotion piece string (e.g. 'wQ') to the single lowercase character expected by chess.js
    const promoChar = piece.length > 1 ? piece[1].toLowerCase() : piece.toLowerCase();

    try {
      const move = activeGame.move({
        from: fromSquare,
        to: toSquare,
        promotion: promoChar,
      });

      if (move === null) {
        setShowPromotionDialog(false);
        setPromotionSquare(null);
        return false;
      }

      const nextGame = new Chess(activeGame.fen());
      setGame(nextGame);
      const status = checkGameStatus(nextGame);

      // Reset promotion modal overlay immediately
      setShowPromotionDialog(false);
      setPromotionSquare(null);

      // Perform deep position evaluation & coaching in the background to keep UI transition completely fluent!
      if (session) {
        (async () => {
          try {
            const nextTensor = boardToTensor(nextGame);
            const inputTensor = new ort.Tensor("float32", nextTensor, [1, 13, 8, 8]);
            const results = await session.run({ input: inputTensor });
            const playedValue = results.value.data[0];

            const bestVal = bestPossibleValueRef.current ?? playedValue;
            const loss = Math.max(0, bestVal - playedValue);

            let classification = "Excellent";
            if (loss <= 0.05) classification = "Excellent";
            else if (loss <= 0.15) classification = "Good";
            else if (loss <= 0.35) classification = "Inaccuracy";
            else if (loss <= 0.65) classification = "Mistake";
            else classification = "Blunder";

            console.log(`🎯 Promotion Move Played: ${move.san} | Eval: ${playedValue.toFixed(2)} | Best possible: ${bestVal.toFixed(2)} | Loss: ${loss.toFixed(4)} | Class: ${classification}`);

            if (onMove) {
              onMove(move.san, playerColor === 'w' ? 'b' : 'w', status.isOver, status.message, loss, classification);
            }
          } catch (e) {
            console.error("Error evaluating promotion move drop classification:", e);
            if (onMove) {
              onMove(move.san, playerColor === 'w' ? 'b' : 'w', status.isOver, status.message, 0.0, "Excellent");
            }
          }
        })();
      } else {
        if (onMove) {
          onMove(move.san, playerColor === 'w' ? 'b' : 'w', status.isOver, status.message, 0.0, "Excellent");
        }
      }

      return true;
    } catch (err) {
      console.warn("Attempted illegal promotion move:", err.message);
      setShowPromotionDialog(false);
      setPromotionSquare(null);
      return false;
    }
  };

  const THEMES = {
    slate: { light: "#EAEFF5", dark: "#64748B" },
    emerald: { light: "#ECECD7", dark: "#739552" },
    ocean: { light: "#DEE3E6", dark: "#5B88A5" },
    wood: { light: "#F0D9B5", dark: "#B58863" }
  };
  const selectedTheme = THEMES[boardThemeName] || THEMES.slate;

  const boardTheme = {
    lightSquare: selectedTheme.light,
    darkSquare: selectedTheme.dark,
    boardWrapper: "rounded-lg overflow-hidden border border-slate-350 shadow-md",
  };

  // Determine checked King highlighting
  const customSquareStyles = {};
  if (game.inCheck()) {
    const kingSquare = getKingSquare(game);
    if (kingSquare) {
      customSquareStyles[kingSquare] = {
        background: "radial-gradient(circle, rgba(239, 68, 68, 0.65) 0%, rgba(220, 38, 38, 0.9) 100%)",
        borderRadius: "4px",
        boxShadow: "inset 0 0 14px rgba(220, 38, 38, 1)"
      };
    }
  }

  return (
    <div className="relative p-2 bg-white/50 border border-slate-250 rounded-xl">
      <Chessboard
        position={game.fen()}
        boardOrientation={playerColor === 'w' ? 'white' : 'black'}
        onPieceDrop={handleDrop}
        customLightSquareStyle={{ backgroundColor: selectedTheme.light }}
        customDarkSquareStyle={{ backgroundColor: selectedTheme.dark }}
        boardWidth={460}
        arePiecesDraggable={gameStarted && !isAiThinking && !isModelLoading}
        customSquareStyles={customSquareStyles}
        onPromotionCheck={(sourceSquare, targetSquare) => {
          const piece = game.get(sourceSquare);
          if (!piece) return false;
          const isPawn = piece.type === 'p';
          const promotionRank = piece.color === 'w' ? '8' : '1';
          if (isPawn && targetSquare[1] === promotionRank) {
            // Strictly enforce chess rules: check if the promotion move is legal using chess.js
            const tempGame = new Chess(game.fen());
            try {
              const move = tempGame.move({ from: sourceSquare, to: targetSquare, promotion: 'q' });
              if (move !== null) {
                setMoveFrom(sourceSquare);
                setMoveTo(targetSquare);
                moveFromRef.current = sourceSquare;
                moveToRef.current = targetSquare;
                setPromotionSquare(targetSquare);
                setShowPromotionDialog(true);
                return true;
              }
            } catch (e) {
              return false;
            }
          }
          return false;
        }}
        showPromotionDialog={showPromotionDialog}
        onPromotionPieceSelect={(piece, promoteFrom, promoteTo) =>
          handlePromotionPieceSelect(piece, promoteFrom, promoteTo)
        }
        promotionToSquare={promotionSquare}
      />
      {game.inCheck() && (
        <div className="absolute top-4 left-1/2 transform -translate-x-1/2 bg-red-600/90 text-white font-black text-xs uppercase tracking-widest px-4 py-1.5 rounded-full shadow-lg border border-red-500 animate-pulse z-10 flex items-center gap-1.5">
          <span>⚠️</span>
          <span>{game.isGameOver() ? "CHECKMATE!" : "CHECK!"}</span>
        </div>
      )}
      {isModelLoading && (
        <div className="absolute inset-0 bg-slate-900/85 flex flex-col justify-center items-center rounded-xl backdrop-blur-sm z-20">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-500 mb-3"></div>
          <p className="text-white text-sm font-medium tracking-wide">Loading Neural Engine...</p>
        </div>
      )}
      {isAiThinking && !isModelLoading && (
        <div className="absolute bottom-4 right-4 bg-blue-600 text-white text-xs font-bold px-3 py-1.5 rounded-full flex items-center gap-1.5 shadow-lg border border-blue-500 animate-pulse z-10">
          <span className="w-1.5 h-1.5 bg-white rounded-full animate-bounce"></span>
          <span>Engine is thinking...</span>
        </div>
      )}
    </div>
  );
}

