const express = require('express');
const cors = require('cors');
const axios = require('axios');  // For making HTTP requests

const app = express();

app.use(cors());
app.use(express.json());

app.post('/api/predict', async (req, res) => {
    const inputData = req.body;

    try {
        const response = await axios.post('YOUR_FLASK_API_URL/predict', inputData);
        res.json(response.data);
    } catch (error) {
        console.error('Error calling the Flask API:', error);
        res.status(500).json({ message: 'Internal server error' });
    }
});

module.exports = app;