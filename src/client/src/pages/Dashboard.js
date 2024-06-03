import React, {useEffect, useState} from 'react';
import { Link } from 'react-router-dom';
import axios from 'axios';
import '../App.css';
import convertDate from "../helpers/date";

function Dashboard() {
  const [validationResult, setValidationResult] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://localhost:8000/latest-validation-result');
        setValidationResult(response.data);
      } catch (error) {
        console.error('Napaka pri pridobivanju rezultata validacije:', error);
      }
    };

    fetchData();
  }, []);

  return (
    <div>
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
          {validationResult && (
            <div>
              <p className="font-bold mb-2">{validationResult.success ? "Validation successful!" : "Validation failed!"}</p>
              <ul className="list-disc ml-8">
                {validationResult.messages.map((message, index) => (
                  <li key={index}>{message}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
