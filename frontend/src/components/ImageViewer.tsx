import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Slider,
  IconButton,
  Typography,
  Paper,
  Grid,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
} from '@mui/material';
import {
  ZoomIn,
  ZoomOut,
  Contrast,
  Brightness6,
  Cached,
} from '@mui/icons-material';
import cornerstone from 'cornerstone-core';
import cornerstoneWADOImageLoader from 'cornerstone-wado-image-loader';
import cornerstoneTools from 'cornerstone-tools';

interface Props {
  imageUrl: string;
  imageType: 'dicom' | 'nifti';
  metadata?: Record<string, any>;
}

const ImageViewer: React.FC<Props> = ({ imageUrl, imageType, metadata }) => {
  const viewerRef = useRef<HTMLDivElement>(null);
  const [windowWidth, setWindowWidth] = useState(400);
  const [windowCenter, setWindowCenter] = useState(200);
  const [zoom, setZoom] = useState(1);
  const [slice, setSlice] = useState(0);
  const [totalSlices, setTotalSlices] = useState(1);
  const [viewMode, setViewMode] = useState<'2d' | '3d'>('2d');

  useEffect(() => {
    // Initialize cornerstone
    if (viewerRef.current) {
      cornerstone.enable(viewerRef.current);
      
      // Load image
      const imageId = imageType === 'dicom' 
        ? `wadouri:${imageUrl}`
        : `nifti:${imageUrl}`;
        
      cornerstone.loadImage(imageId).then(image => {
        cornerstone.displayImage(viewerRef.current!, image);
        
        if (imageType === 'nifti') {
          setTotalSlices(image.data.length);
        }
        
        // Enable tools
        cornerstoneTools.init();
        cornerstoneTools.addTool(cornerstoneTools.WwwcTool);
        cornerstoneTools.addTool(cornerstoneTools.ZoomTool);
        cornerstoneTools.addTool(cornerstoneTools.PanTool);
        cornerstoneTools.setToolActive('Wwwc', { mouseButtonMask: 1 });
      });
    }
    
    return () => {
      if (viewerRef.current) {
        cornerstone.disable(viewerRef.current);
      }
    };
  }, [imageUrl, imageType]);

  const handleWindowingChange = (event: Event, newValue: number | number[]) => {
    if (typeof newValue === 'number') {
      setWindowWidth(newValue);
      if (viewerRef.current) {
        const viewport = cornerstone.getViewport(viewerRef.current);
        viewport.voi.windowWidth = newValue;
        cornerstone.setViewport(viewerRef.current, viewport);
      }
    }
  };

  const handleCenterChange = (event: Event, newValue: number | number[]) => {
    if (typeof newValue === 'number') {
      setWindowCenter(newValue);
      if (viewerRef.current) {
        const viewport = cornerstone.getViewport(viewerRef.current);
        viewport.voi.windowCenter = newValue;
        cornerstone.setViewport(viewerRef.current, viewport);
      }
    }
  };

  const handleZoom = (factor: number) => {
    setZoom(prev => {
      const newZoom = prev * factor;
      if (viewerRef.current) {
        const viewport = cornerstone.getViewport(viewerRef.current);
        viewport.scale = newZoom;
        cornerstone.setViewport(viewerRef.current, viewport);
      }
      return newZoom;
    });
  };

  const handleSliceChange = (event: Event, newValue: number | number[]) => {
    if (typeof newValue === 'number' && imageType === 'nifti') {
      setSlice(newValue);
      if (viewerRef.current) {
        cornerstone.loadImage(`nifti:${imageUrl}#${newValue}`).then(image => {
          cornerstone.displayImage(viewerRef.current!, image);
        });
      }
    }
  };

  const handleViewModeChange = (event: any) => {
    setViewMode(event.target.value);
  };

  return (
    <Paper elevation={3} sx={{ p: 2 }}>
      <Grid container spacing={2}>
        <Grid item xs={12}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
            <Typography variant="h6">Medical Image Viewer</Typography>
            <FormControl size="small">
              <InputLabel>View Mode</InputLabel>
              <Select value={viewMode} onChange={handleViewModeChange}>
                <MenuItem value="2d">2D View</MenuItem>
                <MenuItem value="3d">3D View</MenuItem>
              </Select>
            </FormControl>
          </Box>
        </Grid>
        
        <Grid item xs={12}>
          <Box
            ref={viewerRef}
            sx={{
              width: '100%',
              height: '500px',
              backgroundColor: '#000',
              position: 'relative',
            }}
          />
        </Grid>
        
        <Grid item xs={12}>
          <Box sx={{ display: 'flex', gap: 2, alignItems: 'center' }}>
            <IconButton onClick={() => handleZoom(1.1)}>
              <ZoomIn />
            </IconButton>
            <IconButton onClick={() => handleZoom(0.9)}>
              <ZoomOut />
            </IconButton>
            <IconButton>
              <Contrast />
            </IconButton>
            <IconButton>
              <Brightness6 />
            </IconButton>
            <IconButton onClick={() => cornerstone.reset(viewerRef.current!)}>
              <Cached />
            </IconButton>
          </Box>
        </Grid>
        
        <Grid item xs={6}>
          <Typography gutterBottom>Window Width</Typography>
          <Slider
            value={windowWidth}
            onChange={handleWindowingChange}
            min={1}
            max={1000}
            valueLabelDisplay="auto"
          />
        </Grid>
        
        <Grid item xs={6}>
          <Typography gutterBottom>Window Center</Typography>
          <Slider
            value={windowCenter}
            onChange={handleCenterChange}
            min={-500}
            max={500}
            valueLabelDisplay="auto"
          />
        </Grid>
        
        {imageType === 'nifti' && (
          <Grid item xs={12}>
            <Typography gutterBottom>Slice</Typography>
            <Slider
              value={slice}
              onChange={handleSliceChange}
              min={0}
              max={totalSlices - 1}
              valueLabelDisplay="auto"
            />
          </Grid>
        )}
        
        {metadata && (
          <Grid item xs={12}>
            <Typography variant="h6" gutterBottom>
              Image Metadata
            </Typography>
            <Box sx={{ maxHeight: '200px', overflow: 'auto' }}>
              {Object.entries(metadata).map(([key, value]) => (
                <Typography key={key} variant="body2">
                  <strong>{key}:</strong> {value}
                </Typography>
              ))}
            </Box>
          </Grid>
        )}
      </Grid>
    </Paper>
  );
};

export default ImageViewer; 