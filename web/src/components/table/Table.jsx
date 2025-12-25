import { flexRender } from "@tanstack/react-table";
import React from "react";

const Table = ({ table }) => {
  return (
    <table className="p-2 m-2 rounded-md mx-auto max-w-5xl min-w-sm bg-neutral-100 shadow">
      <thead className="bg-sky-500">
        {table.getHeaderGroups().map((g) => (
          <tr key={g.id}>
            {g.headers.map((h) => (
              <th
                key={h.id}
                className="px-3 py-2 cursor-pointer select-none text-neutral-100"
                onClick={h.column.getToggleSortingHandler()}
              >
                {flexRender(h.column.columnDef.header, h.getContext())}
                {{ asc: " ▲", desc: " ▼" }[h.column.getIsSorted()]}
              </th>
            ))}
          </tr>
        ))}
      </thead>
      <tbody>
        {table.getRowModel().rows.map((row) => (
          <tr key={row.id} className="border-t">
            {row.getVisibleCells().map((cell) => (
              <td key={cell.id} className="px-3 py-2">
                {flexRender(cell.column.columnDef.cell, cell.getContext())}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
};

export default Table;
