import React from "react";
import { Link } from "react-router-dom";
const Home = () => {
  return (
    <div className="text-center rounded-md relative overflow-hidden">
      <img
        src="/public/hero.jpg"
        alt="heart"
        className="w-screen h-screen object-cover"
      />
      <div className="absolute inset-0 bg-black/60"></div>
      <div className="flex absolute inset-0 items-center justify-center">
        <div className="text-neutral-100 text-center">
          <h1 className="text-5xl mb-2">Heart Disease</h1>
          <h4 className="text-2xl mb-2">
            This website is for educational purposes only. For accurate
            detection always visit a doctor.
          </h4>
          <div className="flex flex-row gap-2 justify-center">
            <Link
              className="p-2 rounded-md border hover:bg-sky-500"
              to="/project"
            >
              Project
            </Link>
            <Link
              className="p-2 rounded-md border hover:bg-sky-500"
              to="/new/patient"
            >
              Add Patient
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
