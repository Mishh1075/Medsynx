// frontend/components/UploadForm.tsx

import React, { useState } from 'react';
import axios from 'axios';

const UploadForm = () => {
  const [file, setFile] = useState<File | null>(null);
  const [message, setMessage] = useState('');

  const handleUpload = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    try {
      await axios.post('http://localhost:8000/upload/', formData);
      setMessage('File uploaded successfully. Now you can generate synthetic data.');
    } catch (err) {
      setMessage('Upload failed.');
    }
  };

  return (
    <div>
      <h3>Upload Your CSV</h3>
      <input type="file" accept=".csv" onChange={(e) => setFile(e.target.files?.[0] || null)} />
      <button onClick={handleUpload}>Upload</button>
      <p>{message}</p>
    </div>
  );
};

export default UploadForm;
