# Pneumonia Detection with EfficientNet

This directory contains the server setup to run the EfficientNet pneumonia detection model.

## Overview

The system consists of two main components:

1. **Next.js Frontend** - Upload and view X-ray images for analysis
2. **EfficientNet ML Model** - Flask API serving the trained model

## Setup Instructions

### Prerequisites

- Python 3.8+ with pip
- Node.js 14+ with npm
- The trained EfficientNet model in `pneumonia-ml-efficientnet/output/`

### 1. Start the ML Model Server

From the project root, run:

```bash
# Navigate to the backend directory
cd backend/ml_model

# Start the ML server
python server.py
```

The server will start on http://localhost:5000 by default.

### 2. Configure the Next.js App

Create or edit a `.env.local` file in the project root with:

```
NEXT_PUBLIC_API_URL=http://localhost:5000
```

### 3. Start the Next.js Frontend

From the project root, run:

```bash
# Install dependencies
npm install

# Start the development server
npm run dev
```

Visit http://localhost:3000/dashboard/doctor/upload-xray to use the application.

## Usage

1. Fill in the patient information
2. Upload an X-ray image
3. Click "Analyze X-Ray" to send the image to the EfficientNet model
4. View the analysis results

## Troubleshooting

### "This site can't be reached" Error

If you see "This site can't be reached" when trying to connect to the ML server:

1. **Check if the server is running** - You should see "EfficientNet server started on http://0.0.0.0:5000" in the console
2. **Firewall issues** - Make sure port 5000 is not blocked by your firewall
3. **CORS issues** - The Flask server has CORS enabled, but you might need to add your frontend URL to allowed origins
4. **Networking** - Try accessing the API directly in your browser at http://localhost:5000/

If needed, you can explicitly specify the host and port:

```bash
python server.py --host 0.0.0.0 --port 5000
```

### Other Common Issues

- If the model server shows as offline, check that the Flask server is running
- Make sure the EfficientNet model files are in the correct location (should be in `pneumonia-ml-efficientnet/output/`)
- Check the browser console and server logs for error messages
- Verify the Python environment has all required packages installed (listed in requirements.txt)
- If using a corporate or university network, ensure there are no proxy settings blocking localhost connections 