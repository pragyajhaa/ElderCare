const SerialPort = require('serialport');
const Readline = require('@serialport/parser-readline');
const fs = require('fs');
const express = require('express');

const app = express();
const port = 3000;

// Set up SerialPort to read data from Arduino
const serialPort = new SerialPort('/dev/ttyUSB0', { baudRate: 9600 });  // Update the port to match your setup
const parser = new Readline();
serialPort.pipe(parser);

let sensorData = {};

// Listen to the data from Arduino
parser.on('data', (data) => {
  try {
    // Parse the incoming JSON string from Arduino
    sensorData = JSON.parse(data);

    // Update the JSON file with new data
    fs.writeFileSync('data.json', JSON.stringify(sensorData, null, 2));

    console.log('Updated sensor data:', sensorData);
  } catch (err) {
    console.error('Error parsing data:', err);
  }
});

// Serve the JSON file through an API
app.get('/sensor-data', (req, res) => {
  res.json(sensorData);
});

app.listen(port, () => {
  console.log(`Server running on http://localhost:${port}`);
});
