import React from "react";
import ShikiHighlighter from "react-shiki";

const CodeCard = ({ code }) => {
  return (
    <div className="flex flex-col p-2 m-2 rounded-md mx-auto max-w-5xl min-w-sm justify-center bg-neutral-950 shadow">
      <ShikiHighlighter className="text-sm" language="python" theme="ayu-dark">
        {code}
      </ShikiHighlighter>
    </div>
  );
};

export default CodeCard;
