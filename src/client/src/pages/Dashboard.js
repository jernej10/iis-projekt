import React from 'react';
import { Link } from 'react-router-dom';
import '../App.css';

function Dashboard() {
  return (
    <div className="App">
      <header className="App-header">
        <p>
          Dashboard Page
        </p>
        <Link className="App-link" to="/">
          Home
        </Link>
      </header>
    </div>
  );
}

export default Dashboard;
