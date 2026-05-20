import React from "react";

export default function DifficultySelect({ value, onChange, className }) {
  const levels = ["Easy", "Medium", "Hard"];
  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={`bg-gray-700 text-white rounded p-2 ${className}`}
    >
      {levels.map((lvl) => (
        <option key={lvl} value={lvl}>
          {lvl}
        </option>
      ))}
    </select>
  );
}
