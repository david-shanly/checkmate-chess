import React from "react";

export default function AnalysisPanel({ analysisData, isAiThinking, isModelLoading, theme = "light" }) {
  const { evaluation = 0.0, topMoves = [], liveAccuracy = 100, precision = 100, recall = 100 } = analysisData || {};

  // Format evaluation to display plus/minus signs
  const formatEval = (val) => {
    if (val > 0) return `+${val.toFixed(2)}`;
    if (val < 0) return `${val.toFixed(2)}`;
    return "0.00";
  };

  // Get descriptive status text based on evaluation
  const getEvalStatus = (val) => {
    if (Math.abs(val) <= 0.25) return "Equal position";
    if (val > 0.25 && val <= 1.0) return "Slight advantage for White";
    if (val > 1.0 && val <= 2.5) return "White is winning";
    if (val > 2.5) return "White is completely winning";
    if (val < -0.25 && val >= -1.0) return "Slight advantage for Black";
    if (val < -1.0 && val >= -2.5) return "Black is winning";
    return "Black is completely winning";
  };

  const isDark = theme === "dark";


  return (
    <div className={`border rounded-xl p-5 w-80 font-sans shadow-lg flex flex-col gap-5 h-[480px] transition-all duration-300 ${
      isDark ? "bg-[#1C2538] border-slate-700 text-white" : "bg-white border-slate-350 text-slate-800"
    }`}>
      {/* Header */}
      <div className={`flex items-center justify-between border-b pb-3 transition-all duration-300 ${
        isDark ? "border-slate-700" : "border-slate-200"
      }`}>
        <h2 className={`text-sm font-bold uppercase tracking-wider flex items-center gap-1.5 transition-all duration-300 ${
          isDark ? "text-slate-100" : "text-slate-700"
        }`}>
          🧠 Neural Engine Live
        </h2>
        <div className="flex items-center gap-1.5">
          <span className="relative flex h-2.5 w-2.5">
            {isAiThinking ? (
              <>
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-blue-500"></span>
              </>
            ) : isModelLoading ? (
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-amber-500 animate-pulse"></span>
            ) : (
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-500"></span>
            )}
          </span>
          <span className="text-[10px] font-bold text-slate-400 uppercase tracking-widest">
            {isAiThinking ? "THINKING" : isModelLoading ? "LOADING" : "ACTIVE"}
          </span>
        </div>
      </div>

      {/* 1. Value Head (Evaluation) */}
      <div className={`p-4 rounded-xl border shadow-inner flex flex-col items-center justify-center text-center transition-all duration-300 ${
        isDark ? "bg-[#111A2E] border-slate-700/50" : "bg-[#EAEFF5] border-slate-250/80"
      }`}>
        <span className={`text-[10px] font-extrabold uppercase tracking-widest mb-1 transition-all duration-300 ${
          isDark ? "text-slate-400" : "text-slate-500"
        }`}>
          Value Head (Eval)
        </span>
        <div className={`text-3xl font-extrabold tracking-tight flex items-baseline gap-1 font-sans transition-all duration-300 ${
          isDark ? "text-white" : "text-slate-800"
        }`}>
          {formatEval(evaluation)}
        </div>
        <span className={`text-xs font-bold mt-2 transition-all duration-300 ${
          isDark ? "text-slate-350" : "text-slate-600"
        }`}>
          {getEvalStatus(evaluation)}
        </span>
      </div>

      {/* 2. Policy Head (Candidate Moves strength distribution) */}
      <div className="flex-1 flex flex-col gap-3 min-h-0 overflow-y-auto">
        <div className="flex items-center justify-between">
          <span className={`text-[10px] font-extrabold uppercase tracking-widest transition-all duration-300 ${
            isDark ? "text-slate-400" : "text-slate-500"
          }`}>
            Policy (Top Decisions)
          </span>
          <span className={`text-[9px] px-2 py-0.5 rounded font-bold uppercase tracking-wider border transition-all duration-300 ${
            isDark ? "bg-slate-800 border-slate-700 text-slate-400" : "bg-slate-100 border-slate-200 text-slate-500"
          }`}>
            Model Choice
          </span>
        </div>
        
        <div className="space-y-2.5">
          {topMoves.length === 0 ? (
            <div className="text-xs text-slate-400 italic py-4 text-center">
              Waiting for moves...
            </div>
          ) : (
            topMoves.slice(0, 3).map((candidate, idx) => (
              <div key={idx} className="flex flex-col gap-1">
                <div className="flex justify-between items-center text-xs">
                  <span className={`font-bold px-2 py-0.5 rounded border transition-all duration-300 ${
                    isDark ? "bg-slate-805 border-slate-700 text-slate-200" : "bg-slate-100 border-slate-200 text-slate-700"
                  }`}>
                    {candidate.san}
                  </span>
                  <div className="flex items-center gap-2 font-semibold">
                    <span className="text-slate-400 text-[10px]">({formatEval(candidate.val)})</span>
                    <span className={`font-bold transition-all duration-300 ${
                      isDark ? "text-slate-200" : "text-slate-705"
                    }`}>{Math.round(candidate.prob * 100)}%</span>
                  </div>
                </div>
                {/* Horizontal Progress Bar */}
                <div className={`w-full rounded-full h-1.5 overflow-hidden border transition-all duration-300 ${
                  isDark ? "bg-slate-900 border-slate-800" : "bg-slate-100 border-slate-200"
                }`}>
                  <div
                    className="bg-blue-600 h-full rounded-full transition-all duration-300"
                    style={{ width: `${candidate.prob * 100}%` }}
                  ></div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* 3. Neural Coaching Metrics Dashboard (Accuracy, Precision, Recall) */}
      <div className={`border rounded-xl p-3.5 flex flex-col gap-3 transition-all duration-300 ${
        isDark ? "bg-[#111A2E]/70 border-slate-700/50" : "bg-[#EAEFF5]/80 border-slate-250"
      }`}>
        <div className="flex justify-between items-center px-1">
          <span className="text-[10px] font-extrabold uppercase tracking-widest text-slate-400">
            Neural Precision Metrics
          </span>
          <span className={`text-[9px] px-2 py-0.5 rounded font-extrabold uppercase transition-all duration-300 ${
            liveAccuracy >= 90
              ? "bg-emerald-950/40 text-emerald-400 border border-emerald-800/30"
              : liveAccuracy >= 75
              ? "bg-blue-950/40 text-blue-400 border border-blue-800/30"
              : "bg-amber-950/40 text-amber-400 border border-amber-800/30"
          }`}>
            {liveAccuracy >= 90 ? "Excellent" : liveAccuracy >= 75 ? "Good" : "Fair"}
          </span>
        </div>

        <div className="grid grid-cols-3 gap-2">
          {/* Accuracy Card */}
          <div className={`p-2 rounded-lg border text-center transition-all duration-300 ${
            isDark ? "bg-[#1C2538] border-slate-750" : "bg-white border-slate-200"
          }`}>
            <span className="text-[8px] font-black uppercase text-slate-400 block tracking-wider">
              Accuracy
            </span>
            <div className="text-lg font-black tracking-tight text-blue-500 mt-0.5">
              {Math.round(liveAccuracy)}%
            </div>
          </div>

          {/* Precision Card */}
          <div className={`p-2 rounded-lg border text-center transition-all duration-300 ${
            isDark ? "bg-[#1C2538] border-slate-750" : "bg-white border-slate-200"
          }`}>
            <span className="text-[8px] font-black uppercase text-slate-400 block tracking-wider">
              Precision
            </span>
            <div className="text-lg font-black tracking-tight text-emerald-500 mt-0.5">
              {Math.round(precision)}%
            </div>
          </div>

          {/* Recall Card */}
          <div className={`p-2 rounded-lg border text-center transition-all duration-300 ${
            isDark ? "bg-[#1C2538] border-slate-750" : "bg-white border-slate-200"
          }`}>
            <span className="text-[8px] font-black uppercase text-slate-400 block tracking-wider">
              Recall
            </span>
            <div className="text-lg font-black tracking-tight text-purple-500 mt-0.5">
              {Math.round(recall)}%
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
