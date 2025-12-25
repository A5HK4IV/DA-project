import React, { useMemo } from "react";
import {
  getCoreRowModel,
  getPaginationRowModel,
  getSortedRowModel,
  useReactTable,
} from "@tanstack/react-table";
import { useQuery } from "@tanstack/react-query";
import { getPatients } from "../api/GetPatients";
import Table from "../components/table/Table";
import Paginate from "../components/pagination/Paginate";
const Result = () => {
  const { data = [] } = useQuery({
    queryKey: ["patients"],
    queryFn: getPatients,
  });

  const columns = useMemo(() => [
    { accessorKey: "model", header: "Model" },
    { accessorKey: "age", header: "Patient Age" },
    { accessorKey: "sex", header: "Gender" },
    { accessorKey: "cp", header: "CP" },
    { accessorKey: "trestbps", header: "Trestbps" },
    { accessorKey: "chol", header: "Chol" },
    { accessorKey: "fbs", header: "Fbs" },
    { accessorKey: "restecg", header: "Restecg" },
    { accessorKey: "thalach", header: "thalach" },
    { accessorKey: "exang", header: "Exang" },
    { accessorKey: "oldpeak", header: "Oldpeak" },
    { accessorKey: "slope", header: "Slope" },
    { accessorKey: "ca", header: "CA" },
    { accessorKey: "thal", header: "thal" },
    { accessorKey: "result", header: "Result" },
  ]);

  const table = useReactTable({
    data,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    getPaginationRowModel: getPaginationRowModel(),
    getRowId: (_, index) => index.toString(),
    initialState: { pagination: { pageSize: 10 } },
  });

  return (
    <div className="flex flex-col h-screen items-center justify-center">
      <div className="w-full overflow-x-auto">
        <Table table={table} />
      </div>
      <Paginate table={table} />
    </div>
  );
};

export default Result;
