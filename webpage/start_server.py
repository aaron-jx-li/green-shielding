#!/usr/bin/env python3
"""
Startup script for the annotation server.
This will install dependencies and start the Flask server.
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required Python packages."""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def start_server():
    """Start the Flask server."""
    print("🚀 Starting annotation server...")
    try:
        from server import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except ImportError as e:
        print(f"❌ Error importing server: {e}")
        print("Make sure all dependencies are installed.")
        return False
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🏥 AI Model Response Annotation Tool - Server Startup")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('server.py'):
        print("❌ Error: server.py not found!")
        print("Please run this script from the webpage directory.")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install dependencies. Please install manually:")
        print("   pip install -r requirements.txt")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🌐 Server will be available at: http://localhost:5000")
    print("📊 Health check: http://localhost:5000/health")
    print("📁 Annotations will be saved to: ./annotations/")
    print("=" * 60)
    print("Press Ctrl+C to stop the server")
    print("=" * 60 + "\n")
    
    # Start the server
    start_server()
