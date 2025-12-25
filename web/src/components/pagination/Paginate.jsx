import React from "react";

const Paginate = ({ table }) => {
  return (
    <div className="flex gap-2 items-center justify-center mx-auto min-w-sm max-w-5xl mt-2 ">
      <span>
        Page {table.getState().pagination.pageIndex + 1} of{" "}
        {table.getPageCount()}
      </span>
      <div className="flex flex-row gap-2">
        <button
          className="px-2 py-1 border rounded-md bg-sky-500 hover:bg-sky-700 text-neutral-100 disabled:opacity-50 disabled:bg-sky-300"
          onClick={() => table.previousPage()}
          disabled={!table.getCanPreviousPage()}
        >
          Prev
        </button>
        <button
          className="px-2 py-1 border rounded-md bg-sky-500 hover:bg-sky-700 text-neutral-100 disabled:opacity-50 disabled:bg-sky-300"
          onClick={() => table.nextPage()}
          disabled={!table.getCanNextPage()}
        >
          Next
        </button>
      </div>
    </div>
  );
};

export default Paginate;
