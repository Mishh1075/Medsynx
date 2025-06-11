// frontend/components/GenerateSynthetic.tsx

import React, { useState } from 'react';
import axios from 'axios';

const GenerateSynthetic = () => {
  const [message, setMessage] = useState('');

  const handleGenerate = async () => {
    setMessage('Generating synthetic data...');

    try {
      const response = await axios.post('http://localhost:8000/synthgen/');
      if (response.status === 200) {
        setMessage('Synthetic data generated! You can now download it.');
      }
    } catch (error) {
      setMessage('Generation failed.');
    }
  };

  return (
    <div>
      <button onClick={handleGenerate}>Generate Synthetic Data</button>
      <p>{message}</p>
    </div>
  );
};

export default GenerateSynthetic;
