// frontend/components/DownloadData.tsx

import React from 'react';

const DownloadData = () => {
  return (
    <div>
      <a href="http://localhost:8000/download/" target="_blank" rel="noopener noreferrer">
        <button>Download Synthetic CSV</button>
      </a>
    </div>
  );
};

export default DownloadData;
