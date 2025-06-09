import React, { useState } from 'react';
import {
  Box,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TablePagination,
  Tabs,
  Tab,
  Typography
} from '@mui/material';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

interface Props {
  originalData: any[];
  syntheticData: any[];
  metrics: {
    privacy: any;
    utility: any;
    statistical: any;
  };
}

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

const TabPanel: React.FC<TabPanelProps> = ({ children, value, index }) => (
  <div hidden={value !== index} style={{ padding: 16 }}>
    {value === index && children}
  </div>
);

const DataViewer: React.FC<Props> = ({ originalData, syntheticData, metrics }) => {
  const [tabIndex, setTabIndex] = useState(0);
  const [page, setPage] = useState(0);
  const [rowsPerPage, setRowsPerPage] = useState(10);

  const handleChangePage = (event: unknown, newPage: number) => {
    setPage(newPage);
  };

  const handleChangeRowsPerPage = (event: React.ChangeEvent<HTMLInputElement>) => {
    setRowsPerPage(parseInt(event.target.value, 10));
    setPage(0);
  };

  const handleChangeTab = (event: React.SyntheticEvent, newValue: number) => {
    setTabIndex(newValue);
  };

  const renderDataTable = (data: any[]) => {
    if (!data.length) return null;
    
    const columns = Object.keys(data[0]);
    
    return (
      <TableContainer component={Paper}>
        <Table>
          <TableHead>
            <TableRow>
              {columns.map(col => (
                <TableCell key={col}>{col}</TableCell>
              ))}
            </TableRow>
          </TableHead>
          <TableBody>
            {data
              .slice(page * rowsPerPage, page * rowsPerPage + rowsPerPage)
              .map((row, index) => (
                <TableRow key={index}>
                  {columns.map(col => (
                    <TableCell key={col}>{row[col]}</TableCell>
                  ))}
                </TableRow>
              ))}
          </TableBody>
        </Table>
        <TablePagination
          rowsPerPageOptions={[5, 10, 25]}
          component="div"
          count={data.length}
          rowsPerPage={rowsPerPage}
          page={page}
          onPageChange={handleChangePage}
          onRowsPerPageChange={handleChangeRowsPerPage}
        />
      </TableContainer>
    );
  };

  const renderDistributionCharts = () => {
    const numericalColumns = Object.keys(originalData[0]).filter(
      col => typeof originalData[0][col] === 'number'
    );

    return numericalColumns.map(col => (
      <Box key={col} sx={{ height: 300, mb: 4 }}>
        <Typography variant="h6" gutterBottom>
          {col} Distribution
        </Typography>
        <ResponsiveContainer width="100%" height="100%">
          <LineChart>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey={col} />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line
              data={originalData}
              type="monotone"
              dataKey={col}
              name="Original"
              stroke="#8884d8"
            />
            <Line
              data={syntheticData}
              type="monotone"
              dataKey={col}
              name="Synthetic"
              stroke="#82ca9d"
            />
          </LineChart>
        </ResponsiveContainer>
      </Box>
    ));
  };

  const renderMetricsCharts = () => {
    const privacyData = Object.entries(metrics.privacy).map(([key, value]) => ({
      name: key,
      value: value as number
    }));

    const utilityData = Object.entries(metrics.utility).map(([key, value]) => ({
      name: key,
      value: value as number
    }));

    return (
      <>
        <Box sx={{ height: 300, mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Privacy Metrics
          </Typography>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={privacyData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#8884d8" />
            </BarChart>
          </ResponsiveContainer>
        </Box>

        <Box sx={{ height: 300, mb: 4 }}>
          <Typography variant="h6" gutterBottom>
            Utility Metrics
          </Typography>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={utilityData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#82ca9d" />
            </BarChart>
          </ResponsiveContainer>
        </Box>
      </>
    );
  };

  return (
    <Box sx={{ width: '100%' }}>
      <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tabs value={tabIndex} onChange={handleChangeTab}>
          <Tab label="Original Data" />
          <Tab label="Synthetic Data" />
          <Tab label="Distributions" />
          <Tab label="Metrics" />
        </Tabs>
      </Box>

      <TabPanel value={tabIndex} index={0}>
        {renderDataTable(originalData)}
      </TabPanel>

      <TabPanel value={tabIndex} index={1}>
        {renderDataTable(syntheticData)}
      </TabPanel>

      <TabPanel value={tabIndex} index={2}>
        {renderDistributionCharts()}
      </TabPanel>

      <TabPanel value={tabIndex} index={3}>
        {renderMetricsCharts()}
      </TabPanel>
    </Box>
  );
};

export default DataViewer; 