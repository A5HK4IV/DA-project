import { Route, Routes } from "react-router-dom";
import NewPatient from "./components/forms/NewPatient";
import NavBar from "./components/navbar/NavBar";
import Project from "./views/Project";

function App() {
  return (
    <>
      <NavBar />
      <Routes>
        <Route path="/new/patient" element={<NewPatient />} />
        <Route path="/project" element={<Project />} />
      </Routes>
    </>
  );
}

export default App;
