import React, { useState } from 'react';
import {
  Box,
  Stepper,
  Step,
  StepLabel,
  Button,
  Typography,
  Paper,
  CircularProgress,
  Alert,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import CloudUploadIcon from '@mui/icons-material/CloudUpload';
import SettingsIcon from '@mui/icons-material/Settings';
import AssessmentIcon from '@mui/icons-material/Assessment';
import GetAppIcon from '@mui/icons-material/GetApp';

import PrivacyConfig from './PrivacyConfig';
import DataUpload from './DataUpload';
import GenerationProgress from './GenerationProgress';
import EvaluationResults from './EvaluationResults';

const Input = styled('input')({
  display: 'none',
});

const steps = [
  {
    label: 'Upload Data',
    icon: <CloudUploadIcon />,
  },
  {
    label: 'Configure Privacy',
    icon: <SettingsIcon />,
  },
  {
    label: 'Generate & Evaluate',
    icon: <AssessmentIcon />,
  },
  {
    label: 'Download Results',
    icon: <GetAppIcon />,
  },
];

interface DataFlowProps {
  onComplete?: (results: any) => void;
}

const DataFlow: React.FC<DataFlowProps> = ({ onComplete }) => {
  const [activeStep, setActiveStep] = useState(0);
  const [uploadedData, setUploadedData] = useState<File | null>(null);
  const [privacyConfig, setPrivacyConfig] = useState({
    epsilon: 1.0,
    delta: 1e-5,
    numSamples: 1000,
    noiseMultiplier: 1.0,
  });
  const [generationResults, setGenerationResults] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleNext = async () => {
    setError(null);
    
    try {
      if (activeStep === 0 && !uploadedData) {
        throw new Error('Please upload data first');
      }
      
      if (activeStep === 1) {
        // Validate privacy configuration
        if (privacyConfig.epsilon <= 0) {
          throw new Error('Epsilon must be positive');
        }
        if (privacyConfig.delta <= 0 || privacyConfig.delta >= 1) {
          throw new Error('Delta must be between 0 and 1');
        }
      }
      
      if (activeStep === 2) {
        setLoading(true);
        // Generate synthetic data
        const formData = new FormData();
        formData.append('file', uploadedData as File);
        formData.append('config', JSON.stringify(privacyConfig));
        
        const response = await fetch('/api/generate', {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) {
          throw new Error('Data generation failed');
        }
        
        const results = await response.json();
        setGenerationResults(results);
      }
      
      setActiveStep((prev) => prev + 1);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleBack = () => {
    setActiveStep((prev) => prev - 1);
    setError(null);
  };

  const handleUpload = (file: File) => {
    setUploadedData(file);
    setError(null);
  };

  const handlePrivacyConfigChange = (config: typeof privacyConfig) => {
    setPrivacyConfig(config);
    setError(null);
  };

  const handleDownload = async () => {
    try {
      const response = await fetch('/api/download', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ id: generationResults.id }),
      });
      
      if (!response.ok) {
        throw new Error('Download failed');
      }
      
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'synthetic_data.zip';
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      
      if (onComplete) {
        onComplete(generationResults);
      }
    } catch (err) {
      setError(err.message);
    }
  };

  const renderStepContent = (step: number) => {
    switch (step) {
      case 0:
        return (
          <DataUpload
            onUpload={handleUpload}
            file={uploadedData}
          />
        );
      case 1:
        return (
          <PrivacyConfig
            {...privacyConfig}
            onConfigChange={handlePrivacyConfigChange}
          />
        );
      case 2:
        return (
          <GenerationProgress
            results={generationResults}
            loading={loading}
          />
        );
      case 3:
        return (
          <EvaluationResults
            results={generationResults}
            onDownload={handleDownload}
          />
        );
      default:
        return null;
    }
  };

  return (
    <Box sx={{ width: '100%', mt: 3 }}>
      <Stepper activeStep={activeStep} alternativeLabel>
        {steps.map((step, index) => (
          <Step key={step.label}>
            <StepLabel
              StepIconComponent={() => (
                <Box
                  sx={{
                    width: 40,
                    height: 40,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    borderRadius: '50%',
                    backgroundColor: 
                      activeStep === index
                        ? 'primary.main'
                        : activeStep > index
                        ? 'success.main'
                        : 'grey.300',
                    color: 'white',
                  }}
                >
                  {step.icon}
                </Box>
              )}
            >
              {step.label}
            </StepLabel>
          </Step>
        ))}
      </Stepper>

      <Paper sx={{ p: 3, mt: 3 }}>
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {renderStepContent(activeStep)}

        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 3 }}>
          <Button
            disabled={activeStep === 0}
            onClick={handleBack}
            sx={{ mr: 1 }}
          >
            Back
          </Button>
          <Button
            variant="contained"
            onClick={activeStep === steps.length - 1 ? handleDownload : handleNext}
            disabled={loading}
          >
            {loading ? (
              <CircularProgress size={24} color="inherit" />
            ) : activeStep === steps.length - 1 ? (
              'Download'
            ) : (
              'Next'
            )}
          </Button>
        </Box>
      </Paper>
    </Box>
  );
};

export default DataFlow; 