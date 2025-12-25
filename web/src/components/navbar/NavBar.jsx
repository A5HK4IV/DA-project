import React from "react";
import { Link } from "react-router-dom";
const NavBar = () => {
  return (
    <>
      <nav className="flex justify-center space-x-2">
        <Link
          className="font-medium rounded-lg px-3 py-2 text-gray-700 hover:bg-gray-100 hover:text-gray-900"
          to="/"
        >
          Home
        </Link>
        <Link
          className="font-medium rounded-lg px-3 py-2 text-gray-700 hover:bg-gray-100 hover:text-gray-900"
          to="/project"
        >
          Project
        </Link>
        <Link
          className="font-medium rounded-lg px-3 py-2 text-gray-700 hover:bg-gray-100 hover:text-gray-900"
          to="/new/patient"
        >
          New Patient
        </Link>
        <Link
          className="font-medium rounded-lg px-3 py-2 text-gray-700 hover:bg-gray-100 hover:text-gray-900"
          to="/result"
        >
          Result
        </Link>
      </nav>
    </>
  );
};

export default NavBar;
