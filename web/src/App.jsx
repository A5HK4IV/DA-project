import { Route, Routes } from "react-router-dom";
import NewPatient from "./components/forms/NewPatient";
import NavBar from "./components/navbar/NavBar";
import Project from "./views/Project";
import Result from "./views/Result";
import Home from "./views/Home";

function App() {
  return (
    <>
      <NavBar />
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/new/patient" element={<NewPatient />} />
        <Route path="/project" element={<Project />} />
        <Route path="/result" element={<Result />} />
      </Routes>
    </>
  );
}

export default App;
