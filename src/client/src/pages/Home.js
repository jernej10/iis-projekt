import React from 'react';
import { Link } from 'react-router-dom';
import '../App.css';

function Home() {
  return (
    <div className="App">
      <header className="App-header">
        <p>
          Prediciting S&P 500 stock prices using machine learning
        </p>
        <Link className="App-link" to="/dashboard">
          Dashboard
        </Link>
      </header>
    </div>
  );
}

export default Home;
