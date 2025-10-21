#!/bin/bash

echo "🏥 AI Model Response Annotation Tool - Server Startup"
echo "=================================================="

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found! Please install Python 3."
    exit 1
fi

# Install dependencies
echo "📦 Installing Python dependencies..."
pip3 install -r requirements.txt

# Start the server
echo "🚀 Starting annotation server..."
echo "🌐 Server will be available at: http://localhost:5000"
echo "📁 Annotations will be saved to: ./annotations/"
echo "Press Ctrl+C to stop the server"
echo "=================================================="

python3 server.py
