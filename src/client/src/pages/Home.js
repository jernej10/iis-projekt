import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { Line } from 'react-chartjs-2';
import { Link } from 'react-router-dom';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import '../App.css';

// Register the necessary components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

function Home() {
  const [prediction, setPrediction] = useState(1);
  const [predictionRegression, setPredictionRegression] = useState(0);
  const [loading, setLoading] = useState(true);
  const [loadingRegression, setLoadingRegression] = useState(true);
  const [error, setError] = useState(null);
  const [errorRegression, setErrorRegression] = useState(null);
  const [historicalPrices, setHistoricalPrices] = useState([]);
  const [loadingHistorical, setLoadingHistorical] = useState(true);
  const [errorHistorical, setErrorHistorical] = useState(null);

  useEffect(() => {
    const fetchPredictionWillGoUp = async () => {
      try {
        const response = await axios.get('https://api-production-fb8c.up.railway.app/predict');
        setPrediction(response.data.prediction[0]);
      } catch (error) {
        setError(error);
      } finally {
        setLoading(false);
      }
    };

    const fetchPredictionPrice = async () => {
      try {
        const response = await axios.get('https://api-production-fb8c.up.railway.app/predict/regression');
        setPredictionRegression(response.data.prediction[0]);
      } catch (error) {
        setErrorRegression(error);
      } finally {
        setLoadingRegression(false);
      }
    };

    const fetchHistoricalPrices = async () => {
      try {
        const response = await axios.get('https://api-production-fb8c.up.railway.app/historical-prices');
        setHistoricalPrices(response.data.prices);
      } catch (error) {
        setErrorHistorical(error);
      } finally {
        setLoadingHistorical(false);
      }
    };

    fetchPredictionWillGoUp();
    fetchPredictionPrice();
    fetchHistoricalPrices();
  }, []);

  const chartData = {
    labels: historicalPrices.map(price => price.Date),
    datasets: [
      {
        label: 'S&P 500 Price (USD)',
        data: historicalPrices.map(price => price.Close),
        fill: false,
        borderColor: '#61dafb',
        tension: 0.1,
      },
    ],
  };

  return (
    <div className="App">
      <header className="App-header">
        <p>Predicting sp500 stock price 📈</p>
        <Link className="App-link" to="/dashboard">
          Dashboard
        </Link>
      </header>
      <h1 className="text-header">STOCK PRICE FOR TOMORROW</h1>
      {loadingRegression && <p className="text-white">Loading price prediction for tomorrow...</p>}
      {errorRegression && <p>Error: {error.message}</p>}
      {!loadingRegression && !error && (
        <h2 className="price">{predictionRegression}$</h2>
      )}
      <div className="flex justify-center items-center h-[40vh]">
      {loading && <p className="text-white">Loading prediction will price go up tomorrow...</p>}
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
      <h2 className="text-header">Historical S&P 500 Prices</h2>
      {loadingHistorical && <p className="text-white">Loading historical prices...</p>}
      {errorHistorical && <p>Error: {error.message}</p>}
      {!loadingHistorical && !errorHistorical && (
          <div className="flex justify-center items-center h-[40vh] mb-12">
          <Line data={chartData} />
          </div>
      )}
    </div>
  );
}

export default Home;
