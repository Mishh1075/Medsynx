import React, { Component, ErrorInfo, ReactNode } from 'react';
import { Alert, Button, Container, Typography, Box } from '@mui/material';
import { ErrorOutline } from '@mui/icons-material';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null
  };

  public static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error, errorInfo: null };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    this.setState({
      error: error,
      errorInfo: errorInfo
    });
    
    // Log error to monitoring service
    console.error('Uncaught error:', error, errorInfo);
  }

  private handleReset = () => {
    this.setState({
      hasError: false,
      error: null,
      errorInfo: null
    });
  };

  public render() {
    if (this.state.hasError) {
      return (
        <Container maxWidth="sm">
          <Box
            sx={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              gap: 2,
              mt: 4
            }}
          >
            <ErrorOutline color="error" sx={{ fontSize: 64 }} />
            
            <Typography variant="h5" component="h1" gutterBottom>
              Something went wrong
            </Typography>
            
            <Alert severity="error" sx={{ width: '100%' }}>
              {this.state.error && this.state.error.toString()}
            </Alert>
            
            {this.state.errorInfo && (
              <Box
                sx={{
                  width: '100%',
                  maxHeight: '200px',
                  overflow: 'auto',
                  bgcolor: 'grey.100',
                  p: 2,
                  borderRadius: 1
                }}
              >
                <pre style={{ margin: 0 }}>
                  {this.state.errorInfo.componentStack}
                </pre>
              </Box>
            )}
            
            <Box sx={{ mt: 2 }}>
              <Button
                variant="contained"
                color="primary"
                onClick={this.handleReset}
              >
                Try Again
              </Button>
            </Box>
          </Box>
        </Container>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary; 