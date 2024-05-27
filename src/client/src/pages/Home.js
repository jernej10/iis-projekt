import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Link } from 'react-router-dom';
import '../App.css';

function Home() {
  const [prediction, setPrediction] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchPrediction = async () => {
      try {
        const response = await axios.get('http://localhost:8000/predict');
        setPrediction(response.data.prediction[0]);
      } catch (error) {
        setError(error);
      } finally {
        setLoading(false);
      }
    };

    fetchPrediction();
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <p>Predicting S&P 500 stock price for tomorrow using machine learning</p>
        <Link className="App-link" to="/dashboard">
          Dashboard
        </Link>
      </header>
      <h1 className="text-header">STOCK PRICE FOR TOMORROW</h1>
      <h2 className="price">{"200"}€</h2>
      <div className="container">
        {loading && <p className="text-white">Loading prediction for tomorrow...</p>}
        {error && <p>Error: {error.message}</p>}
        {!loading && !error && (
          <div
            className={`prediction-container ${prediction === 1 ? 'up' : 'down'}`}
          >
            <div className="arrow">
              {prediction === 1 ? '↑' : '↓'}
            </div>
            <div className="text">
              {prediction === 1 ? 'Stock will go UP tomorrow' : 'Stock will go DOWN tomorrow'}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default Home;
