import React from 'react';
import {
  Box,
  CircularProgress,
  LinearProgress,
  Typography,
  Paper,
  Stepper,
  Step,
  StepLabel,
  Alert
} from '@mui/material';

export interface ProgressStep {
  label: string;
  status: 'pending' | 'in_progress' | 'completed' | 'error';
  progress?: number;
  message?: string;
}

interface Props {
  steps: ProgressStep[];
  currentStep: number;
  title?: string;
  error?: string;
}

const ProgressFeedback: React.FC<Props> = ({
  steps,
  currentStep,
  title = 'Operation in Progress',
  error
}) => {
  const activeStep = steps.findIndex(step => step.status === 'in_progress');
  const hasError = error || steps.some(step => step.status === 'error');

  return (
    <Paper sx={{ p: 3, maxWidth: 600, mx: 'auto', mt: 4 }}>
      <Box sx={{ textAlign: 'center', mb: 3 }}>
        <Typography variant="h6" gutterBottom>
          {title}
        </Typography>
        
        {!hasError && (
          <CircularProgress
            size={24}
            sx={{ mb: 2, visibility: activeStep >= 0 ? 'visible' : 'hidden' }}
          />
        )}
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 3 }}>
          {error}
        </Alert>
      )}

      <Stepper activeStep={currentStep} orientation="vertical">
        {steps.map((step, index) => (
          <Step key={step.label} completed={step.status === 'completed'}>
            <StepLabel error={step.status === 'error'}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                <Typography>{step.label}</Typography>
                {step.status === 'in_progress' && (
                  <CircularProgress size={16} />
                )}
              </Box>
            </StepLabel>
            
            {step.status === 'in_progress' && step.progress !== undefined && (
              <Box sx={{ mt: 1, mx: 2 }}>
                <LinearProgress
                  variant="determinate"
                  value={step.progress}
                  sx={{ height: 8, borderRadius: 4 }}
                />
                <Typography variant="caption" sx={{ mt: 0.5, display: 'block' }}>
                  {`${Math.round(step.progress)}%`}
                </Typography>
              </Box>
            )}
            
            {step.message && (
              <Typography
                variant="caption"
                color={step.status === 'error' ? 'error' : 'textSecondary'}
                sx={{ mt: 0.5, display: 'block', ml: 2 }}
              >
                {step.message}
              </Typography>
            )}
          </Step>
        ))}
      </Stepper>
    </Paper>
  );
};

export default ProgressFeedback; 