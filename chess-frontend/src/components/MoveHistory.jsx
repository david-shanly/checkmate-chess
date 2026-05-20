import React, { useEffect, useRef } from "react";

export default function MoveHistory({ moves, onDownloadPgn, theme = "light" }) {
  const containerRef = useRef(null);

  // Auto-scroll to bottom when new moves are added
  useEffect(() => {
    if (containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [moves]);

  // Group flat moves array into pairs (White, Black)
  const pairs = [];
  for (let i = 0; i < moves.length; i += 2) {
    pairs.push({
      num: Math.floor(i / 2) + 1,
      white: moves[i],
      black: moves[i + 1] || "",
    });
  }

  const isDark = theme === "dark";

  return (
    <div className={`border rounded-xl overflow-hidden w-80 font-sans shadow-lg flex flex-col h-[480px] transition-all duration-300 ${
      isDark ? "bg-[#1C2538] border-slate-700" : "bg-white border-slate-350"
    }`}>
      {/* Scrollable table content area */}
      <div 
        ref={containerRef}
        className="flex-1 overflow-y-auto min-h-0"
      >
        <table className="w-full text-left border-collapse table-fixed">
          <thead className={`sticky top-0 shadow-sm z-10 border-b transition-all duration-300 ${
            isDark ? "bg-[#111A2E] border-slate-700" : "bg-[#DCE6F1] border-slate-300"
          }`}>
            <tr className={`text-xs font-bold uppercase tracking-wider transition-all duration-300 ${
              isDark ? "text-slate-300" : "text-slate-700"
            }`}>
              <th className={`py-2.5 px-3 border-r w-12 text-center transition-all duration-300 ${
                isDark ? "border-slate-700" : "border-slate-300"
              }`}>#</th>
              <th className={`py-2.5 px-4 border-r transition-all duration-300 ${
                isDark ? "border-slate-700" : "border-slate-300"
              }`}>White</th>
              <th className="py-2.5 px-4">Black</th>
            </tr>
          </thead>
          <tbody className={`divide-y text-sm font-medium transition-all duration-300 ${
            isDark ? "divide-slate-800 text-slate-200" : "divide-slate-100 text-slate-800"
          }`}>
            {pairs.length === 0 ? (
              <tr>
                <td colSpan="3" className="py-16 text-center text-slate-400 italic">
                  No moves played yet
                </td>
              </tr>
            ) : (
              pairs.map((pair) => (
                <tr key={pair.num} className={`transition-colors ${
                  isDark ? "hover:bg-slate-800/40" : "hover:bg-slate-50"
                }`}>
                  <td className={`py-2.5 px-3 border-r text-center font-bold text-xs transition-all duration-300 ${
                    isDark ? "border-slate-800 bg-slate-900/30 text-slate-500" : "border-slate-100 bg-slate-50/50 text-slate-400"
                  }`}>
                    {pair.num}
                  </td>
                  <td className={`py-2.5 px-4 border-r font-semibold transition-all duration-300 ${
                    isDark ? "border-slate-800 text-slate-100" : "border-slate-100 text-slate-800"
                  }`}>
                    {pair.white}
                  </td>
                  <td className={`py-2.5 px-4 font-semibold ${
                    isDark ? "text-slate-100" : "text-slate-800"
                  }`}>
                    {pair.black}
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Download PGN footer */}
      <div className={`border-t p-2.5 flex justify-center transition-all duration-300 ${
        isDark ? "bg-slate-900/40 border-slate-800" : "bg-slate-50 border-slate-200"
      }`}>
        <button
          onClick={onDownloadPgn}
          disabled={moves.length === 0}
          className="w-full bg-blue-600 hover:bg-blue-500 disabled:bg-slate-200 disabled:text-slate-400 text-white font-bold py-2 px-4 rounded-lg text-xs transition duration-150 flex items-center justify-center gap-1.5 shadow-sm active:scale-[0.98]"
        >
          📥 Download PGN
        </button>
      </div>
    </div>
  );
}


