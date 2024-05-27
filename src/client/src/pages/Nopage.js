import '../App.css';
import {Link} from "react-router-dom";
import React from "react";

const NoPage = () => {
  return (
    <div className="App">
      <header className="App-header">
        <p>
          404 | Page Not Found
        </p>
        <Link className="App-link" to="/">
          Home
        </Link>
      </header>
    </div>
  );};

export default NoPage;