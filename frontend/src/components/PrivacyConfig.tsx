import React from 'react';
import {
  Box,
  Paper,
  Typography,
  Slider,
  TextField,
  Tooltip,
  IconButton,
  Alert,
} from '@mui/material';
import InfoIcon from '@mui/icons-material/Info';

interface PrivacyConfigProps {
  epsilon: number;
  delta: number;
  numSamples: number;
  noiseMultiplier: number;
  onConfigChange: (config: {
    epsilon: number;
    delta: number;
    numSamples: number;
    noiseMultiplier: number;
  }) => void;
}

const PrivacyConfig: React.FC<PrivacyConfigProps> = ({
  epsilon,
  delta,
  numSamples,
  noiseMultiplier,
  onConfigChange,
}) => {
  const handleEpsilonChange = (event: Event, newValue: number | number[]) => {
    if (typeof newValue === 'number') {
      onConfigChange({
        epsilon: newValue,
        delta,
        numSamples,
        noiseMultiplier,
      });
    }
  };

  const handleDeltaChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseFloat(event.target.value);
    if (!isNaN(newValue)) {
      onConfigChange({
        epsilon,
        delta: newValue,
        numSamples,
        noiseMultiplier,
      });
    }
  };

  const handleNumSamplesChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const newValue = parseInt(event.target.value);
    if (!isNaN(newValue)) {
      onConfigChange({
        epsilon,
        delta,
        numSamples: newValue,
        noiseMultiplier,
      });
    }
  };

  const handleNoiseMultiplierChange = (event: Event, newValue: number | number[]) => {
    if (typeof newValue === 'number') {
      onConfigChange({
        epsilon,
        delta,
        numSamples,
        noiseMultiplier: newValue,
      });
    }
  };

  return (
    <Paper sx={{ p: 3, mb: 3 }}>
      <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Typography variant="h6">Privacy Parameters</Typography>
        <Tooltip title="These parameters control the privacy guarantees of the synthetic data generation process.">
          <IconButton size="small" sx={{ ml: 1 }}>
            <InfoIcon />
          </IconButton>
        </Tooltip>
      </Box>

      <Alert severity="info" sx={{ mb: 3 }}>
        Adjusting these parameters affects the privacy-utility trade-off of the generated data.
      </Alert>

      <Box sx={{ mb: 3 }}>
        <Typography gutterBottom>
          Epsilon (ε) - Privacy Budget
          <Tooltip title="Lower values provide stronger privacy guarantees but may reduce utility">
            <IconButton size="small">
              <InfoIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Typography>
        <Slider
          value={epsilon}
          onChange={handleEpsilonChange}
          min={0.1}
          max={10}
          step={0.1}
          marks={[
            { value: 0.1, label: '0.1' },
            { value: 1, label: '1' },
            { value: 5, label: '5' },
            { value: 10, label: '10' },
          ]}
          valueLabelDisplay="auto"
        />
      </Box>

      <Box sx={{ mb: 3 }}>
        <Typography gutterBottom>
          Delta (δ) - Privacy Relaxation
          <Tooltip title="Probability of privacy breach, should be very small">
            <IconButton size="small">
              <InfoIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Typography>
        <TextField
          type="number"
          value={delta}
          onChange={handleDeltaChange}
          inputProps={{
            step: '1e-6',
            min: '1e-10',
            max: '1e-4',
          }}
          fullWidth
        />
      </Box>

      <Box sx={{ mb: 3 }}>
        <Typography gutterBottom>
          Number of Synthetic Samples
          <Tooltip title="Number of synthetic records to generate">
            <IconButton size="small">
              <InfoIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Typography>
        <TextField
          type="number"
          value={numSamples}
          onChange={handleNumSamplesChange}
          inputProps={{
            step: 100,
            min: 100,
            max: 100000,
          }}
          fullWidth
        />
      </Box>

      <Box>
        <Typography gutterBottom>
          Noise Multiplier
          <Tooltip title="Controls the amount of noise added for privacy">
            <IconButton size="small">
              <InfoIcon fontSize="small" />
            </IconButton>
          </Tooltip>
        </Typography>
        <Slider
          value={noiseMultiplier}
          onChange={handleNoiseMultiplierChange}
          min={0.1}
          max={2}
          step={0.1}
          marks={[
            { value: 0.1, label: '0.1' },
            { value: 1, label: '1' },
            { value: 2, label: '2' },
          ]}
          valueLabelDisplay="auto"
        />
      </Box>
    </Paper>
  );
};

export default PrivacyConfig; 