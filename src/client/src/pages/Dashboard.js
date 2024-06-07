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
import convertDate from "../helpers/date";

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

function Dashboard() {
  // State variables for validationResult, metrics, and their loading states
  const [validationResult, setValidationResult] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [classificationMetrics, setClassificationMetrics] = useState(null);
  const [validationResultLoading, setValidationResultLoading] = useState(true);
  const [metricsLoading, setMetricsLoading] = useState(true);
  const [metricLimit, setMetricLimit] = useState(0);
  const [metricLimitLoading, setMetricLimitLoading] = useState(true);

  useEffect(() => {
    const fetchValidationResult = async () => {
      try {
        const response = await axios.get('https://api-production-fb8c.up.railway.app/latest-validation-result');
        setValidationResult(response.data);
      } catch (error) {
        console.error('Error fetching validation result:', error);
      } finally {
        setValidationResultLoading(false);
      }
    };

    const fetchMetrics = async () => {
      try {
        const response = await axios.get('https://api-production-fb8c.up.railway.app/metrics-history');
        const data = response.data;
        const METRICS_TO_SHOW = 5;
        data.classification = data.classification.slice(-METRICS_TO_SHOW);
        data.regression = data.regression.slice(-METRICS_TO_SHOW);
        setMetrics(data);
      } catch (error) {
        console.error('Error fetching metrics:', error);
      } finally {
        setMetricsLoading(false);
      }
    };

    const fetchMetricLimit = async () => {
      try {
        const response = await axios.get('https://api-production-fb8c.up.railway.app/metric-limit/latest');
        console.log(response.data.latest_metric_limit)
        setMetricLimit(parseFloat(response.data.latest_metric_limit.value));
      } catch (error) {
        console.error('Error fetching metric limit:', error);
      } finally {
        setMetricLimitLoading(false);
      }
    };

    const fetchProductionMetrics = async () => {
      try {
        const response = await axios.get('https://api-production-fb8c.up.railway.app/production-metrics-history');
        setClassificationMetrics(response.data.classification);
      } catch (error) {
        console.error('Error fetching production metrics:', error);
      }
    };

    fetchValidationResult();
    fetchMetrics();
    fetchMetricLimit();
    fetchProductionMetrics();
  }, []);

  const handleMetricLimitChange = (e) => {
    setMetricLimit(e.target.value);
  };

  const handleMetricLimitSubmit = async (e) => {
    e.preventDefault();
    try {
      await axios.post('https://api-production-fb8c.up.railway.app/metric-limit', { value: parseFloat(metricLimit) });
      alert('Metric limit updated successfully!');
      setMetricLimit(metricLimit)
    } catch (error) {
      console.error('Error updating metric limit:', error);
      alert('Failed to update metric limit.');
    }
  };

  const data = metrics ? {
    labels: metrics.regression.map((_, index) => `Run ${index + 1}`),
    datasets: [
      {
        label: 'Mean Absolute Error',
        data: metrics.regression.map(metric => metric.mae),
        borderColor: 'rgba(75, 192, 192, 1)',
        backgroundColor: 'rgba(75, 192, 192, 0.2)',
        fill: true
      }
    ]
  } : {};

  return (
    <div className="mb-24">
      <header className="App-header text-center">
        <p>
          Dashboard Page 📆
        </p>
        <Link className="App-link" to="/">
          Home
        </Link>
      </header>
      <div className="mt-8">
        <div className={"mx-8"}>
          <h1 className="font-bold text-2xl mb-4">Data testing results [{validationResult && convertDate(validationResult.timestamp)}] 📝</h1>
          <a href="https://stalwart-praline-262fa2.netlify.app/data_drift.html" className="block underline mb-4" target="_blank" rel="noopener noreferrer">
            ➡️ Data Drift
          </a>
          <a href="https://stalwart-praline-262fa2.netlify.app/stability_tests.html" className="block underline" target="_blank" rel="noopener noreferrer">
            ➡️ Stability Test
          </a>
          <h1 className="font-bold text-2xl my-4">Data validation results [{validationResult && convertDate(validationResult.timestamp)}] 📊</h1>
          {validationResultLoading ? ( // Display loading message while loading
            <p>Loading validation results...</p>
          ) : (
            validationResult && (
              <div>
                <p className="font-bold mb-2">{validationResult.success ? "Validation successful ✅" : "Validation failed ❌"}</p>
                <ul className="list-none ml-8">
                  {validationResult.messages.map((message, index) => (
                    <li key={index}>{message}</li>
                  ))}
                </ul>
              </div>
            )
          )}
          <h1 className="font-bold text-2xl my-4">Metrics History 📜</h1>
          {metricsLoading ? (
            <p>Loading metrics history...</p>
          ) : (
            metrics && (
              <div>
                <div className="chart-container" style={{ width: '500px', height: '300px' }}>
                  <Line data={data} />
                </div>
                <div className="flex">
                  <div>
                    <h2 className="font-bold text-xl mt-4 mb-2">Classification Metrics</h2>
                    <ul className="list-none divide-y-4">
                      {metrics.classification.map((metric, index) => (
                        <li key={index} className="my-2">
                          <p>Accuracy: {metric.accuracy}, Precision: {metric.precision}, Recall: {metric.recall}</p>
                        </li>
                      ))}
                    </ul>
                  </div>
                  <div className="ml-12">
                    <h2 className="font-bold text-xl mt-4 mb-2">Regression Metrics</h2>
                    <ul className="list-none divide-y-4">
                      {metrics.regression.map((metric, index) => (
                        <li key={index} className="my-2">
                          <p>MSE: {metric.mse}, MAE: {metric.mae}, EVS: {metric.evs}</p>
                        </li>
                      ))}
                    </ul>
                  </div>
                </div>
              </div>
            )
          )}
          <div className="mb-4">
            <h1 className="font-bold text-2xl my-4">Production Classification Metrics 🤖</h1>
            {classificationMetrics && (
              <div>
                <ul className="list-none divide-y-4">
                  {classificationMetrics.map((metric, index) => (
                    <li key={index} className="my-2">
                      <p>Accuracy: {metric.accuracy}, Precision: {metric.precision}, Recall: {metric.recall}, F1: {metric.f1}</p>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
          <div className="mb-6">
            <h1 className="font-bold text-2xl my-2">Production Accuracy Limit 🚨</h1>
            <p className="mb-4 text-gray-400">(i) We will send you an email if accuracy in production falls below the limit.</p>
            {metricLimitLoading ? (
              <p>Loading metric limit...</p>
            ) : (
              <form onSubmit={handleMetricLimitSubmit} className="w-60">
                <div className="mb-4">
                  <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="value">
                    Accuracy
                  </label>
                  <input
                    className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
                    id="value"
                    type="number"
                    name="value"
                    value={metricLimit}
                    onChange={handleMetricLimitChange}
                  />
                </div>
                <button
                  className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
                  type="submit"
                >
                  Update Limit
                </button>
              </form>
            )}
          </div>
          <div>
            <h1 className="font-bold text-2xl my-4">Interpretability of models 📊</h1>
            <h2 className="font-bold text-xl mt-4 mb-2">Regression</h2>
            <img src="http://127.0.0.1:8000/img/beeswarm_regression.png" alt="Regression" style={{ maxWidth: '100%', maxHeight: '300px' }} />
          </div>
          <div>
            <h2 className="font-bold text-xl mt-4 mb-2">Classification</h2>
            <img src="http://127.0.0.1:8000/img/beeswarm_classification.png" alt="Classification" style={{ maxWidth: '100%', maxHeight: '300px' }} />
          </div>
          <div className="mt-4">
            <h2 className="font-bold mb-2">💡 Legend:</h2>
            <p>f0 - Close, f1 - Volume, f2 - Open, f3 - High, f4 - Low, f5 - Open_Nasdaq</p>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
