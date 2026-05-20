import React, { useState, useEffect, useRef } from "react";
import ChessBoard from "./components/ChessBoard";
import MoveHistory from "./components/MoveHistory";
import AnalysisPanel from "./components/AnalysisPanel";

// Custom Chess.com-style Evaluation Bar
function EvaluationBar({ evaluation }) {
  // Convert evaluation to percentage from White's perspective
  // Sigmoid scaling: percentage = 100 / (1 + exp(-0.8 * eval))
  const percentage = 100 / (1 + Math.exp(-0.8 * evaluation));
  const clippedPercentage = Math.max(5, Math.min(95, percentage));

  return (
    <div className="relative w-5 h-[460px] bg-[#2b2b2b] rounded-md overflow-hidden border border-slate-700 shadow-lg flex flex-col justify-end">
      {/* White side (bottom part) */}
      <div 
        className="bg-white transition-all duration-500 ease-out"
        style={{ height: `${clippedPercentage}%` }}
      ></div>
      
      {/* Live evaluation score text overlay */}
      <div className={`absolute left-0 right-0 text-center font-bold text-[9px] select-none pointer-events-none transition-all duration-300 ${
        evaluation >= 0 ? 'bottom-2 text-slate-900' : 'top-2 text-white'
      }`}>
        {evaluation >= 0 ? `+${evaluation.toFixed(1)}` : `${evaluation.toFixed(1)}`}
      </div>
    </div>
  );
}

export default function App() {
  const [theme, setTheme] = useState(() => localStorage.getItem("chess-theme") || "dark");
  const [gamesPlayed, setGamesPlayed] = useState(() => Number(localStorage.getItem("chess-games-played")) || 0);
  const [hasGameBeenCounted, setHasGameBeenCounted] = useState(false);
  const [pastAccuracies, setPastAccuracies] = useState(() => {
    const stored = localStorage.getItem("chess-game-accuracies");
    return stored ? JSON.parse(stored) : [];
  });

  const [gameStarted, setGameStarted] = useState(false);
  const [startDifficulty, setStartDifficulty] = useState(() => localStorage.getItem("chess-start-difficulty") || "1400");
  const [timeLimit, setTimeLimit] = useState(10); // in minutes
  const [history, setHistory] = useState([]);
  const [playerTime, setPlayerTime] = useState(600); // White in seconds
  const [engineTime, setEngineTime] = useState(600); // Black in seconds
  const [activeTurn, setActiveTurn] = useState("w"); // "w" | "b"
  const [gameOverMessage, setGameOverMessage] = useState(null);
  const [resetTrigger, setResetTrigger] = useState(0);

  // States for dynamic playing color & board customization
  const [chosenColor, setChosenColor] = useState(() => localStorage.getItem("chess-chosen-color") || "w");
  const [activePlayerColor, setActivePlayerColor] = useState("w");
  const [boardTheme, setBoardTheme] = useState(() => localStorage.getItem("chess-board-theme") || "slate");

  // Dynamic analysis & precision coaching state
  const [analysisData, setAnalysisData] = useState({ evaluation: 0.0, topMoves: [], liveAccuracy: 100, precision: 100, recall: 100 });
  const [moveStats, setMoveStats] = useState([]); // [{ moveNum, san, loss, classification }]

  // Custom Elo Input States
  const [showCustomInput, setShowCustomInput] = useState(false);
  const [customDifficultyInput, setCustomDifficultyInput] = useState("");

  const timerIntervalRef = useRef(null);

  const isCalibrated = pastAccuracies.length > 0;
  const averageAccuracy = isCalibrated
    ? Math.round(
        pastAccuracies.reduce((sum, item) => {
          const acc = typeof item === "number" ? item : item.accuracy;
          return sum + acc;
        }, 0) / pastAccuracies.length
      )
    : 0;

  // Stabilized skill Elo logic: starting skill level serves as baseline guess until calibrated.
  // Once calibrated, the estimated Elo is based strictly on actual gameplay history,
  // factoring in accuracy, opponent Elo, and the game outcome (win/loss/draw)!
  const estimatedElo = (() => {
    if (!isCalibrated) return Number(startDifficulty);
    
    let weightedSum = 0;
    let totalWeight = 0;
    
    pastAccuracies.forEach((item, index) => {
      let opp = 1500;
      let acc = 50;
      let outcome = "draw";
      if (typeof item === "number") {
        acc = item;
        outcome = acc >= 70 ? "win" : acc <= 40 ? "loss" : "draw"; // fallback for legacy data
      } else if (item && typeof item === "object") {
        acc = item.accuracy;
        opp = item.opponentRating || 1500;
        outcome = item.outcome || (acc >= 70 ? "win" : acc <= 40 ? "loss" : "draw"); // fallback
      }
      
      // Calculate performance rating:
      // Base value shifts by outcome (+150 for win, -150 for loss, +0 for draw)
      // Refined by accuracy relative to average (accuracy - 50) * 4
      let outcomeModifier = 0;
      if (outcome === "win") outcomeModifier = 150;
      else if (outcome === "loss") outcomeModifier = -150;
      
      const gamePerformance = opp + outcomeModifier + (acc - 50) * 4;
      
      // Weight increases linearly with game index: oldest has weight 1, latest has weight N
      const weight = index + 1;
      weightedSum += gamePerformance * weight;
      totalWeight += weight;
    });
    
    return Math.round(weightedSum / totalWeight);
  })();

  const getRecommendedDifficulty = (elo) => {
    if (elo < 800) return "600";
    if (elo < 1000) return "900";
    if (elo < 1200) return "1100";
    if (elo < 1600) return "1400";
    if (elo < 2000) return "1800";
    if (elo < 2200) return "2100";
    if (elo < 2300) return "2250";
    if (elo < 2400) return "2350";
    if (elo < 2500) return "2450";
    return "2600";
  };

  // Resolve the actual difficulty level passed to the engine
  const getEngineDifficulty = () => {
    // If they choose one of the dynamic calibrated options:
    const userEloStr = estimatedElo.toString();
    const belowEloStr = Math.max(1000, estimatedElo - 100).toString();
    const aboveEloStr = Math.min(2500, estimatedElo + 100).toString();

    if (startDifficulty === userEloStr) {
      return getRecommendedDifficulty(estimatedElo);
    }
    if (startDifficulty === belowEloStr) {
      return getRecommendedDifficulty(Math.max(1000, estimatedElo - 100));
    }
    if (startDifficulty === aboveEloStr) {
      return getRecommendedDifficulty(Math.min(2500, estimatedElo + 100));
    }

    // Otherwise, play at EXACTLY the chosen fixed difficulty
    return getRecommendedDifficulty(Number(startDifficulty));
  };

  const activeDifficulty = getEngineDifficulty();

  const handleStartDifficultyChange = (val) => {
    if (val === "custom_input") {
      setShowCustomInput(true);
    } else {
      setShowCustomInput(false);
      setStartDifficulty(val);
      localStorage.setItem("chess-start-difficulty", val);
    }
  };

  const handleApplyCustomDifficulty = () => {
    const elo = Number(customDifficultyInput);
    if (!elo || elo < 1000 || elo > 2500) {
      alert("Please enter a valid rating between 1000 and 2500.");
      return;
    }

    const val = elo.toString();
    setStartDifficulty(val);
    localStorage.setItem("chess-start-difficulty", val);
    localStorage.setItem("chess-custom-difficulty", val);
    setShowCustomInput(false);
    setCustomDifficultyInput("");
  };

  const handleResetStats = () => {
    localStorage.removeItem("chess-games-played");
    localStorage.removeItem("chess-game-accuracies");
    localStorage.removeItem("chess-start-difficulty");
    localStorage.removeItem("chess-custom-difficulty");
    setGamesPlayed(0);
    setPastAccuracies([]);
    setStartDifficulty("1500");
    setShowCustomInput(false);
  };

  const toggleTheme = () => {
    setTheme((t) => {
      const nextTheme = t === "light" ? "dark" : "light";
      localStorage.setItem("chess-theme", nextTheme);
      return nextTheme;
    });
  };

  const triggerGameCompletion = (finalAccuracy, outcome = "draw") => {
    if (!hasGameBeenCounted) {
      setHasGameBeenCounted(true);
      setGamesPlayed((prev) => {
        const newCount = prev + 1;
        localStorage.setItem("chess-games-played", newCount);
        return newCount;
      });

      if (finalAccuracy !== undefined) {
        setPastAccuracies((prev) => {
          // Record accuracy, opponent rating, and game outcome for advanced calibration
          const gameRecord = { 
            accuracy: finalAccuracy, 
            opponentRating: Number(startDifficulty),
            outcome 
          };
          const updated = [...prev, gameRecord];
          localStorage.setItem("chess-game-accuracies", JSON.stringify(updated));
          return updated;
        });
      }
    }
  };

  // 1️⃣ Game Loop (Timers)
  useEffect(() => {
    if (gameStarted && !gameOverMessage) {
      timerIntervalRef.current = setInterval(() => {
        if (activeTurn === activePlayerColor) {
          setPlayerTime((t) => {
            if (t <= 1) {
              clearInterval(timerIntervalRef.current);
              setGameOverMessage("Time out! Neural Engine wins!");
              triggerGameCompletion(35, "loss"); // Low accuracy baseline for timed out games
              return 0;
            }
            return t - 1;
          });
        } else {
          setEngineTime((t) => {
            if (t <= 1) {
              clearInterval(timerIntervalRef.current);
              setGameOverMessage("Time out! You win!");
              triggerGameCompletion(85, "win"); // High accuracy baseline if AI times out
              return 0;
            }
            return t - 1;
          });
        }
      }, 1000);
    }

    return () => {
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gameStarted, activeTurn, gameOverMessage, activePlayerColor]);

  // 2️⃣ Handle Move Event from Board
  const handleMove = (san, nextTurn, isGameOver, gameOverReason, loss, classification) => {
    setHistory((h) => [...h, san]);
    setActiveTurn(nextTurn);

    // If it's a move played by the player, record and evaluate accuracy statistics
    if (loss !== undefined && classification !== undefined) {
      setMoveStats((stats) => {
        const newStats = [...stats, { moveNum: Math.floor(history.length / 2) + 1, san, loss, classification }];
        
        // Dynamic live accuracy updater
        const averageLoss = newStats.reduce((sum, s) => sum + s.loss, 0) / newStats.length;
        const liveAccuracy = Math.max(10, Math.min(100, 100 * Math.exp(-2 * averageLoss)));
        
        // Calculate Precision: High-Quality moves (Best, Excellent, Brilliant, Good) divided by total
        const totalPlayerMoves = newStats.length;
        const highQualityMoves = newStats.filter(s => 
          s.classification === "Best" || 
          s.classification === "Excellent" || 
          s.classification === "Brilliant" || 
          s.classification === "Good"
        ).length;
        
        // Calculate Recall: Absolute Top Decisions (Best or Brilliant) matching played move
        const bestMoves = newStats.filter(s => 
          s.classification === "Best" || 
          s.classification === "Brilliant"
        ).length;

        const precision = totalPlayerMoves > 0 ? (highQualityMoves / totalPlayerMoves) * 100 : 100;
        const recall = totalPlayerMoves > 0 ? (bestMoves / totalPlayerMoves) * 100 : 100;

        setAnalysisData((prev) => ({
          ...prev,
          liveAccuracy,
          precision,
          recall
        }));

        if (isGameOver) {
          if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
          setGameOverMessage(gameOverReason);
          
          let outcome = "draw";
          if (gameOverReason.includes("(You)") || gameOverReason.includes("You win")) {
            outcome = "win";
          } else if (gameOverReason.includes("Neural Engine") || gameOverReason.includes("Engine wins")) {
            outcome = "loss";
          }
          triggerGameCompletion(Math.round(liveAccuracy), outcome);
        }

        return newStats;
      });
    } else {
      if (isGameOver) {
        if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
        setGameOverMessage(gameOverReason);
        
        let outcome = "draw";
        if (gameOverReason.includes("(You)") || gameOverReason.includes("You win")) {
          outcome = "win";
        } else if (gameOverReason.includes("Neural Engine") || gameOverReason.includes("Engine wins")) {
          outcome = "loss";
        }
        triggerGameCompletion(Math.round(analysisData.liveAccuracy), outcome);
      }
    }
  };

  // 3️⃣ Real-Time Engine Analysis update callback
  const handleAnalysisUpdate = (data) => {
    // Retain existing accuracy and precision metrics while updating engine candidate evaluations
    setAnalysisData((prev) => ({
      ...data,
      liveAccuracy: prev.liveAccuracy,
      precision: prev.precision,
      recall: prev.recall
    }));
  };

  // 4️⃣ Start Game Handler
  const handleStartGame = () => {
    setHistory([]);
    setMoveStats([]);
    setAnalysisData({ evaluation: 0.0, topMoves: [], liveAccuracy: 100 });
    setPlayerTime(timeLimit * 60);
    setEngineTime(timeLimit * 60);
    
    // Resolve dynamic player color assignment (w, b, or random)
    let assignedColor = chosenColor;
    if (chosenColor === "random") {
      assignedColor = Math.random() < 0.5 ? "w" : "b";
    }
    setActivePlayerColor(assignedColor);

    setActiveTurn("w"); // White always moves first
    setGameOverMessage(null);
    setHasGameBeenCounted(false);
    setResetTrigger((prev) => prev + 1);
    setGameStarted(true);
  };

  // 5️⃣ Reset & Exit to Main Menu
  const handleExit = () => {
    if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
    setGameStarted(false);
    setGameOverMessage(null);
    setHistory([]);
    setMoveStats([]);
  };

  // 6️⃣ Resign Handler
  const handleResign = () => {
    if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
    triggerGameCompletion(Math.round(analysisData.liveAccuracy), "loss");
    setGameOverMessage("You resigned! Neural Engine wins!");
  };

  // 7️⃣ Portable Game Notation (PGN) Export Generator
  const downloadPGN = () => {
    if (history.length === 0) return;

    // Compile PGN pairs
    const pairs = [];
    for (let i = 0; i < history.length; i += 2) {
      const white = history[i];
      const black = history[i + 1] || "";
      pairs.push(`${Math.floor(i / 2) + 1}. ${white} ${black}`);
    }

    const pgnHeaders = [
      `[Event "Player vs Neural Engine"]`,
      `[Site "Chess Trainer App"]`,
      `[Date "${new Date().toISOString().slice(0, 10).replace(/-/g, ".")}"]`,
      `[Round "1"]`,
      `[White "You"]`,
      `[Black "Neural Engine"]`,
      `[Result "${gameOverMessage ? (gameOverMessage.includes("You win") ? "1-0" : gameOverMessage.includes("Engine wins") ? "0-1" : "1/2-1/2") : "*"}"]`,
      `[Difficulty "${activeDifficulty}"]`,
      `[TimeControl "${timeLimit * 60}"]`,
      `[LiveAccuracy "${Math.round(analysisData.liveAccuracy)}%"]`,
      `\n`
    ].join("\n");

    const pgnText = pgnHeaders + pairs.join(" ") + ` ${gameOverMessage ? (gameOverMessage.includes("You win") ? "1-0" : gameOverMessage.includes("Engine wins") ? "0-1" : "1/2-1/2") : "*"}`;

    const blob = new Blob([pgnText], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `chess-trainer-${new Date().toISOString().slice(0, 10)}.pgn`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Compile final game report metrics
  const getGameAccuracyReport = () => {
    if (moveStats.length === 0) {
      return { excellent: 0, good: 0, inaccuracy: 0, mistake: 0, blunder: 0, accuracy: 100, coachText: "No moves evaluated." };
    }

    const excellent = moveStats.filter((s) => s.classification === "Excellent").length;
    const good = moveStats.filter((s) => s.classification === "Good").length;
    const inaccuracy = moveStats.filter((s) => s.classification === "Inaccuracy").length;
    const mistake = moveStats.filter((s) => s.classification === "Mistake").length;
    const blunder = moveStats.filter((s) => s.classification === "Blunder").length;
    
    const averageLoss = moveStats.reduce((sum, s) => sum + s.loss, 0) / moveStats.length;
    const accuracy = Math.round(Math.max(10, Math.min(100, 100 * Math.exp(-2 * averageLoss))));

    let coachText = "Perfect game! Unbelievable chess accuracy. You outplayed the Neural Engine perfectly!";
    if (blunder > 0) {
      coachText = "A good battle, but the blunders were costly. Review your tactical decisions under pressure!";
    } else if (mistake > 0) {
      coachText = "A solid game, but a few critical mistakes turned the tide. Analyze key moves to improve!";
    } else if (inaccuracy > 0) {
      coachText = "Great accuracy! You played with excellent strategic sense, only dropping minor evaluation points.";
    }

    return { excellent, good, inaccuracy, mistake, blunder, accuracy, coachText };
  };

  // Format seconds to MM:SS
  const formatTime = (timeInSeconds) => {
    const m = Math.floor(timeInSeconds / 60).toString().padStart(2, "0");
    const s = (timeInSeconds % 60).toString().padStart(2, "0");
    return `${m}:${s}`;
  };

  const startingLevels = (() => {
    const levels = [
      { val: "600", label: "600 (Beginner)" },
      { val: "900", label: "900 (Novice)" },
      { val: "1100", label: "1100 (Casual Player)" },
      { val: "1400", label: "1400 (Intermediate)" },
      { val: "1800", label: "1800 (Advanced Club Player)" },
      { val: "2100", label: "2100 (Expert / Candidate Master)" },
      { val: "2250", label: "2250 (National Master)" },
      { val: "2350", label: "2350 (FIDE Master FM)" },
      { val: "2450", label: "2450 (International Master IM)" },
      { val: "2600", label: "2600 (Grandmaster GM)" },
    ];

    // Inject any custom difficulty rating they have typed/saved!
    const savedCustom = localStorage.getItem("chess-custom-difficulty");
    if (savedCustom) {
      if (!levels.some((l) => l.val === savedCustom)) {
        levels.push({
          val: savedCustom,
          label: `${savedCustom} (Custom Rating 🧠)`
        });
      }
    }

    if (isCalibrated) {
      // 1. Dynamic Tier slightly below calibrated skill (-100 Elo)
      const belowElo = Math.max(1000, estimatedElo - 100);
      const belowEloStr = belowElo.toString();
      if (belowElo !== estimatedElo && !levels.some((l) => l.val === belowEloStr)) {
        levels.push({
          val: belowEloStr,
          label: `${belowEloStr} (Slightly Below My Skill 📉)`
        });
      }

      // 2. Exact Calibrated Skill Elo
      const userEloStr = estimatedElo.toString();
      if (!levels.some((l) => l.val === userEloStr)) {
        levels.push({
          val: userEloStr,
          label: `${userEloStr} (My Calibrated Skill 🧠)`
        });
      }

      // 3. Dynamic Tier slightly above calibrated skill (+100 Elo)
      const aboveElo = Math.min(2500, estimatedElo + 100);
      const aboveEloStr = aboveElo.toString();
      if (aboveElo !== estimatedElo && !levels.some((l) => l.val === aboveEloStr)) {
        levels.push({
          val: aboveEloStr,
          label: `${aboveEloStr} (Slightly Above My Skill 📈)`
        });
      }
    }

    // Add a marker for Custom option
    levels.push({ val: "custom_input", label: "✨ Enter Custom Rating..." });

    // Sort by rating value (except the custom option)
    levels.sort((a, b) => {
      if (a.val === "custom_input") return 1;
      if (b.val === "custom_input") return -1;
      return Number(a.val) - Number(b.val);
    });

    return levels;
  })();

  const isDark = theme === "dark";

  // Render Start Screen
  if (!gameStarted) {
    return (
      <div className={`min-h-screen flex flex-col items-center justify-center p-4 transition-all duration-300 ${
        isDark ? "bg-[#0B111E]" : "bg-slate-100"
      }`}>
        <div className="w-full max-w-xl flex justify-between items-center mb-4">
          {isCalibrated ? (
            <button
              onClick={handleResetStats}
              className={`font-bold py-2 px-4 rounded-xl text-xs transition-all duration-300 shadow-md flex items-center gap-1.5 border active:scale-95 ${
                isDark 
                  ? "bg-red-950/40 hover:bg-red-900/60 border-red-900/50 text-red-300" 
                  : "bg-red-50 hover:bg-red-100 border-red-200 text-red-700"
              }`}
            >
              🧹 Clear Stats
            </button>
          ) : (
            <div></div>
          )}
          <button
            onClick={toggleTheme}
            className={`font-bold py-2 px-4 rounded-xl text-xs transition-all duration-300 shadow-md flex items-center gap-1.5 border active:scale-95 ${
              isDark 
                ? "bg-[#1C2538] hover:bg-slate-800 border-slate-700 text-white" 
                : "bg-white hover:bg-slate-50 border-slate-250 text-slate-700"
            }`}
          >
            {isDark ? "☀️ Light Theme" : "🌙 Dark Theme"}
          </button>
        </div>

        <div className={`rounded-xl shadow-2xl p-8 w-full max-w-xl border transition-all duration-300 ${
          isDark ? "bg-[#1C2538] border-slate-700" : "bg-[#EAEFF5] border-slate-350"
        }`}>
          <div className="text-center mb-8">
            <h1 className={`text-4xl font-extrabold tracking-tight mb-2 font-sans transition-all duration-300 ${
              isDark ? "text-white" : "text-slate-800"
            }`}>
              Chess Trainer
            </h1>
            <p className="text-blue-650 font-bold text-lg tracking-wide uppercase font-sans">
              Just checkin' mate!
            </p>
          </div>

          <div className="space-y-6">
            <div>
              <label className={`block font-bold text-sm uppercase mb-2 tracking-wide font-sans transition-all duration-300 ${
                isDark ? "text-slate-400" : "text-slate-650"
              }`}>
                Starting Skill Rating
              </label>
              <select
                value={showCustomInput ? "custom_input" : startDifficulty}
                onChange={(e) => handleStartDifficultyChange(e.target.value)}
                className={`w-full font-medium border rounded-lg p-3 text-base outline-none focus:ring-2 focus:ring-blue-500 transition shadow-sm cursor-pointer ${
                  isDark ? "bg-[#111A2E] text-white border-slate-700" : "bg-white text-slate-805 border-slate-300"
                }`}
              >
                {startingLevels.map((lvl) => (
                  <option key={lvl.val} value={lvl.val}>
                    {lvl.label}
                  </option>
                ))}
              </select>

              {showCustomInput && (
                <div className="mt-3 flex gap-2 animate-fadeIn">
                  <input
                    type="number"
                    min="1000"
                    max="2500"
                    placeholder="Enter rating (1000 - 2500)"
                    value={customDifficultyInput}
                    onChange={(e) => setCustomDifficultyInput(e.target.value)}
                    className={`flex-1 font-medium border rounded-lg p-2.5 text-sm outline-none focus:ring-2 focus:ring-blue-500 transition ${
                      isDark ? "bg-[#111A2E] text-white border-slate-700" : "bg-white text-slate-805 border-slate-300"
                    }`}
                  />
                  <button
                    onClick={handleApplyCustomDifficulty}
                    className="bg-blue-600 hover:bg-blue-500 text-white font-bold px-4 py-2 rounded-lg text-xs transition duration-150 active:scale-95 shadow-md"
                  >
                    Apply
                  </button>
                </div>
              )}
            </div>

            <div>
              <label className={`block font-bold text-sm uppercase mb-2 tracking-wide font-sans transition-all duration-300 ${
                isDark ? "text-slate-400" : "text-slate-650"
              }`}>
                Time Control
              </label>
              <select
                value={timeLimit}
                onChange={(e) => setTimeLimit(Number(e.target.value))}
                className={`w-full font-medium border rounded-lg p-3 text-base outline-none focus:ring-2 focus:ring-blue-500 transition shadow-sm cursor-pointer ${
                  isDark ? "bg-[#111A2E] text-white border-slate-700" : "bg-white text-slate-805 border-slate-300"
                }`}
              >
                <option value={5}>5 Minutes</option>
                <option value={10}>10 Minutes</option>
                <option value={15}>15 Minutes</option>
              </select>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className={`block font-bold text-sm uppercase mb-2 tracking-wide font-sans transition-all duration-300 ${
                  isDark ? "text-slate-400" : "text-slate-650"
                }`}>
                  Play As
                </label>
                <div className="flex rounded-lg overflow-hidden border border-slate-300 dark:border-slate-700 shadow-sm">
                  {[
                    { val: "w", label: "⬜ White" },
                    { val: "b", label: "⬛ Black" },
                    { val: "random", label: "🎲 Random" }
                  ].map((opt) => (
                    <button
                      key={opt.val}
                      onClick={() => {
                        setChosenColor(opt.val);
                        localStorage.setItem("chess-chosen-color", opt.val);
                      }}
                      className={`flex-1 font-bold py-2.5 text-xs transition duration-150 active:scale-95 ${
                        chosenColor === opt.val
                          ? "bg-blue-600 text-white"
                          : isDark
                            ? "bg-[#111A2E] text-slate-300 hover:bg-slate-800"
                            : "bg-white text-slate-700 hover:bg-slate-50"
                      }`}
                    >
                      {opt.label}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className={`block font-bold text-sm uppercase mb-2 tracking-wide font-sans transition-all duration-300 ${
                  isDark ? "text-slate-400" : "text-slate-650"
                }`}>
                  Board Style
                </label>
                <select
                  value={boardTheme}
                  onChange={(e) => {
                    setBoardTheme(e.target.value);
                    localStorage.setItem("chess-board-theme", e.target.value);
                  }}
                  className={`w-full font-medium border rounded-lg p-2.5 text-sm outline-none focus:ring-2 focus:ring-blue-500 transition shadow-sm cursor-pointer ${
                    isDark ? "bg-[#111A2E] text-white border-slate-700" : "bg-white text-slate-805 border-slate-300"
                  }`}
                >
                  <option value="slate">🩶 Sleek Slate</option>
                  <option value="emerald">💚 Emerald Tournament</option>
                  <option value="ocean">💙 Deep Ocean</option>
                  <option value="wood">🤎 Traditional Wood</option>
                </select>
              </div>
            </div>
            {/* Neural Skill Matching Engine Progression Tracker */}
            {isCalibrated && (
              <div className={`border p-4 rounded-xl text-center shadow-inner transition-all duration-300 ${
                isDark ? "bg-slate-900/60 border-slate-750" : "bg-white/60 border-slate-250"
              }`}>
                <span className="text-[10px] font-extrabold uppercase tracking-widest text-slate-500">
                  🧠 Neural Skill Matching Engine
                </span>
                
                <div className="flex flex-col items-center justify-center mt-3 mb-2">
                  <div className="text-3xl font-black text-blue-600 tracking-tight">
                    {estimatedElo} Elo
                  </div>
                  <div className={`text-xs font-bold px-2.5 py-0.5 rounded-full mt-1 border shadow-sm ${
                    isDark ? "bg-slate-800 border-slate-700 text-slate-300" : "bg-white border-slate-200 text-slate-650"
                  }`}>
                    {!isCalibrated ? "Starting Skill Level" : "Autocalibrated Skill Rating"}
                  </div>
                </div>

                <div className={`grid grid-cols-2 gap-2 text-left text-[11px] font-semibold mt-2 border-t pt-3 ${
                  isDark ? "border-slate-800 text-slate-400" : "border-slate-200 text-slate-650"
                }`}>
                  <div>
                    Matches Played: <span className="font-extrabold text-blue-600">{gamesPlayed}</span>
                  </div>
                  <div>
                    Your Avg Accuracy: <span className="font-extrabold text-blue-600">{isCalibrated ? `${averageAccuracy}%` : "N/A"}</span>
                  </div>
                </div>

                <p className="text-[10px] text-slate-500 mt-3 italic">
                  {!isCalibrated 
                    ? `Matchmaking initialized closer to your choice: currently set to ${activeDifficulty} rating.` 
                    : `Neural Engine skill level has been automatically adjusted to ${activeDifficulty} rating based on your gameplay!`}
                </p>
              </div>
            )}

            <button
              onClick={handleStartGame}
              className="w-full bg-blue-600 hover:bg-blue-500 active:scale-[0.98] text-white font-bold text-lg py-3.5 px-6 rounded-xl transition duration-150 flex items-center justify-center gap-2 shadow-lg shadow-blue-600/20"
            >
              <span>▶</span> Start Game
            </button>
          </div>
        </div>
      </div>
    );
  }

  const accuracyReport = getGameAccuracyReport();

  // Render Play Screen (3-column layout)
  return (
    <div className={`min-h-screen flex flex-col items-center justify-center p-6 font-sans select-none transition-all duration-300 ${
      isDark ? "bg-[#0B111E]" : "bg-slate-100"
    }`}>
      <div className="w-full max-w-6xl flex flex-col">
        {/* Top Timer Bar */}
        <div className={`text-white py-3 px-6 flex justify-between items-center rounded-t-xl border-t border-x shadow-lg select-none transition-all duration-300 ${
          isDark ? "bg-[#1C2538] border-slate-700" : "bg-[#1C2538] border-slate-300"
        }`}>
          <div className="flex items-center gap-3">
            <div className={`w-3 h-3 rounded-full ${activeTurn === activePlayerColor ? "bg-green-500 animate-pulse shadow-lg shadow-green-500/50" : "bg-slate-600"}`}></div>
            <span className="font-bold text-slate-100 tracking-wide">You ({activePlayerColor === "w" ? "White" : "Black"}) ({formatTime(playerTime)})</span>
          </div>
          
          <div className="flex items-center gap-2.5">
            <div className="text-xs bg-slate-800 text-slate-400 px-3 py-1 rounded-full uppercase tracking-wider font-bold border border-slate-700">
               Skill: {activeDifficulty} (🧠 Adaptive)
            </div>
            <button
              onClick={toggleTheme}
              className="bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white font-bold py-1 px-2.5 rounded text-[10px] uppercase tracking-wider border border-slate-700 transition active:scale-95"
            >
              {isDark ? "☀️ Light" : "🌙 Dark"}
            </button>
          </div>

          <div className="flex items-center gap-3">
            <span className="font-bold text-slate-100 tracking-wide">Neural Engine ({activePlayerColor === "w" ? "Black" : "White"}) ({formatTime(engineTime)})</span>
            <div className={`w-3 h-3 rounded-full ${activeTurn !== activePlayerColor ? "bg-green-500 animate-pulse shadow-lg shadow-green-500/50" : "bg-slate-600"}`}></div>
          </div>
        </div>

        {/* Board & Move History Play Panel (3 Column Layout) */}
        <div className={`p-6 rounded-b-xl border flex flex-col lg:flex-row gap-6 items-start justify-center shadow-2xl transition-all duration-300 ${
          isDark ? "bg-[#111A2E] border-slate-700" : "bg-[#EAEFF5] border-slate-3-to-4 border-slate-300"
        }`}>
          {/* Column 1: Board & Eval Bar */}
          <div className={`flex gap-4 items-center p-3 rounded-xl shadow-inner w-full lg:w-auto justify-center border transition-all duration-300 ${
            isDark ? "bg-slate-900/40 border-slate-700/50" : "bg-white/40 border-slate-250"
          }`}>
            <EvaluationBar evaluation={analysisData.evaluation} />
            <ChessBoard
              onMove={handleMove}
              onAnalysisUpdate={handleAnalysisUpdate}
              difficulty={activeDifficulty}
              gameStarted={gameStarted}
              resetTrigger={resetTrigger}
              playerColor={activePlayerColor}
              boardThemeName={boardTheme}
            />
          </div>

          {/* Column 2: Move History & controls */}
          <div className="flex flex-col gap-4 w-full lg:w-auto items-center">
            <MoveHistory moves={history} onDownloadPgn={downloadPGN} theme={theme} />

            <div className="flex gap-3 w-full">
              <button
                onClick={handleResign}
                className="flex-1 bg-red-650 hover:bg-red-600 text-white font-bold py-2.5 px-4 rounded-lg transition active:scale-95 text-sm shadow-md border border-red-700 flex items-center justify-center gap-1.5"
              >
                🏳️ Resign
              </button>
              <button
                onClick={handleExit}
                className="flex-1 bg-slate-700 hover:bg-slate-600 text-white font-bold py-2.5 px-4 rounded-lg transition active:scale-95 text-sm shadow-md border border-slate-650 flex items-center justify-center gap-1.5"
              >
                🏠 Main Menu
              </button>
            </div>
          </div>

          {/* Column 3: Live Neural Engine Analysis Panel */}
          <div className="w-full lg:w-auto flex justify-center">
            <AnalysisPanel
              analysisData={analysisData}
              isAiThinking={activeTurn === "b" && !gameOverMessage}
              isModelLoading={analysisData.topMoves.length === 0 && history.length === 0}
              theme={theme}
            />
          </div>
        </div>
      </div>

      {/* Game Over Popup Modal with AI Coach report card */}
      {gameOverMessage && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 animate-fade-in backdrop-blur-sm">
          <div className={`border p-8 rounded-2xl max-w-md w-full text-center shadow-2xl mx-4 transform scale-100 transition-all duration-300 flex flex-col gap-5 ${
            isDark ? "bg-[#1C2538] border-slate-700 text-white" : "bg-white border-slate-350 text-slate-800"
          }`}>
            <div>
              <div className="text-5xl mb-2">🏆</div>
              <h2 className={`text-2xl font-bold font-sans transition-all duration-300 ${
                isDark ? "text-slate-100" : "text-slate-800"
              }`}>Game Over</h2>
              <p className={`text-sm mt-1 transition-all duration-300 ${
                isDark ? "text-slate-400" : "text-slate-500"
              }`}>{gameOverMessage}</p>
            </div>

            {/* AI Coaching Report Card */}
            <div className={`rounded-xl p-4 border flex flex-col gap-4 text-left transition-all duration-300 ${
              isDark ? "bg-slate-900/60 border-slate-850/30" : "bg-[#EAEFF5]/60 border-slate-250/50"
            }`}>
              <div className={`flex items-center gap-4 border-b pb-3 transition-all duration-300 ${
                isDark ? "border-slate-800" : "border-slate-250"
              }`}>
                {/* Visual Circular Accuracy display */}
                <div className="relative w-16 h-16 rounded-full border-4 border-blue-500/30 flex items-center justify-center font-black text-xl text-blue-500 shadow-inner bg-blue-500/5">
                  {accuracyReport.accuracy}%
                </div>
                <div className="flex-1">
                  <span className="text-[10px] font-extrabold uppercase tracking-widest text-slate-500">
                    AI ACCURACY COACH
                  </span>
                  <p className={`text-xs font-semibold mt-1 italic transition-all duration-300 ${
                    isDark ? "text-slate-350" : "text-slate-600"
                  }`}>
                    "{accuracyReport.coachText}"
                  </p>
                </div>
              </div>

              {/* Move Classifications breakdown grid */}
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className={`flex justify-between items-center p-2 rounded border transition-all duration-300 ${
                  isDark ? "bg-slate-800/40 border-slate-800" : "bg-white border-slate-200"
                }`}>
                  <span className="text-emerald-500 font-bold">✨ Excellent:</span>
                  <span className={`font-bold ${isDark ? "text-slate-250" : "text-slate-700"}`}>{accuracyReport.excellent}</span>
                </div>
                <div className={`flex justify-between items-center p-2 rounded border transition-all duration-300 ${
                  isDark ? "bg-slate-800/40 border-slate-800" : "bg-white border-slate-200"
                }`}>
                  <span className="text-blue-500 font-bold">👍 Good:</span>
                  <span className={`font-bold ${isDark ? "text-slate-250" : "text-slate-700"}`}>{accuracyReport.good}</span>
                </div>
                <div className={`flex justify-between items-center p-2 rounded border transition-all duration-300 ${
                  isDark ? "bg-slate-800/40 border-slate-800" : "bg-white border-slate-200"
                }`}>
                  <span className="text-yellow-600 font-bold">⚠️ Inaccuracies:</span>
                  <span className={`font-bold ${isDark ? "text-slate-250" : "text-slate-700"}`}>{accuracyReport.inaccuracy}</span>
                </div>
                <div className={`flex justify-between items-center p-2 rounded border transition-all duration-300 ${
                  isDark ? "bg-slate-800/40 border-slate-800" : "bg-white border-slate-200"
                }`}>
                  <span className="text-orange-500 font-bold">❌ Mistakes:</span>
                  <span className={`font-bold ${isDark ? "text-slate-250" : "text-slate-700"}`}>{accuracyReport.mistake}</span>
                </div>
                <div className={`flex justify-between items-center p-2 rounded border col-span-2 transition-all duration-300 ${
                  isDark ? "bg-slate-800/40 border-slate-800 animate-pulse" : "bg-white border-slate-200"
                }`}>
                  <span className="text-red-500 font-bold">💥 Blunders:</span>
                  <span className={`font-bold ${isDark ? "text-slate-250" : "text-slate-700"}`}>{accuracyReport.blunder}</span>
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="flex flex-col gap-2.5">
              <button
                onClick={downloadPGN}
                className="w-full bg-emerald-600 hover:bg-emerald-500 text-white font-bold py-3 px-6 rounded-xl transition duration-150 active:scale-95 shadow-md flex items-center justify-center gap-1.5"
              >
                📥 Download PGN
              </button>
              <button
                onClick={handleStartGame}
                className="w-full bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 px-6 rounded-xl transition duration-150 active:scale-95 shadow-md"
              >
                🔄 Play Again (Same Settings)
              </button>
              <button
                onClick={handleExit}
                className={`w-full font-bold py-3 px-6 rounded-xl transition duration-150 active:scale-95 shadow-md border ${
                  isDark 
                    ? "bg-[#1C2538] hover:bg-slate-800 border-slate-700 text-white" 
                    : "bg-white hover:bg-slate-50 border-slate-250 text-slate-700"
                }`}
              >
                🏠 Return to Main Menu
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}


