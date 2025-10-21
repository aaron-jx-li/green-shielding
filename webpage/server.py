#!/usr/bin/env python3
"""
Flask server for handling annotation saves.
This server receives annotation data from the web interface and saves it to local files.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import csv
import os
from datetime import datetime
import pandas as pd

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Create data directory if it doesn't exist
DATA_DIR = 'annotations'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

@app.route('/')
def serve_index():
    """Serve the main HTML file."""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS, JSON)."""
    return send_from_directory('.', filename)

@app.route('/save_annotations', methods=['POST'])
def save_annotations():
    """Save annotation data to local files."""
    try:
        # Get the annotation data from the request
        annotation_data = request.get_json()
        
        if not annotation_data:
            return jsonify({'error': 'No data received'}), 400
        
        # Create timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSON
        json_filename = f"{DATA_DIR}/annotations_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, indent=2, ensure_ascii=False)
        
        # Save as CSV
        csv_filename = f"{DATA_DIR}/annotations_{timestamp}.csv"
        df = pd.DataFrame(annotation_data)
        df.to_csv(csv_filename, index=False, encoding='utf-8')
        
        # Also save to a "latest" file for easy access
        latest_json = f"{DATA_DIR}/latest_annotations.json"
        latest_csv = f"{DATA_DIR}/latest_annotations.csv"
        
        with open(latest_json, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, indent=2, ensure_ascii=False)
        
        df.to_csv(latest_csv, index=False, encoding='utf-8')
        
        print(f"✅ Saved annotations to:")
        print(f"   - {json_filename}")
        print(f"   - {csv_filename}")
        print(f"   - {latest_json}")
        print(f"   - {latest_csv}")
        
        return jsonify({
            'success': True,
            'message': f'Annotations saved successfully!',
            'files_created': [json_filename, csv_filename, latest_json, latest_csv],
            'timestamp': timestamp
        })
        
    except Exception as e:
        print(f"❌ Error saving annotations: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_annotations', methods=['GET'])
def get_annotations():
    """Get the latest annotation data."""
    try:
        latest_file = f"{DATA_DIR}/latest_annotations.json"
        if os.path.exists(latest_file):
            with open(latest_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return jsonify({'success': True, 'data': data})
        else:
            return jsonify({'success': False, 'message': 'No annotations found'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/list_annotation_files', methods=['GET'])
def list_annotation_files():
    """List all annotation files."""
    try:
        files = []
        if os.path.exists(DATA_DIR):
            for filename in os.listdir(DATA_DIR):
                if filename.endswith(('.json', '.csv')):
                    filepath = os.path.join(DATA_DIR, filename)
                    stat = os.stat(filepath)
                    files.append({
                        'filename': filename,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    })
        
        return jsonify({'success': True, 'files': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'message': 'Annotation server is running'})

if __name__ == '__main__':
    print("🚀 Starting Annotation Server...")
    print(f"📁 Annotations will be saved to: {os.path.abspath(DATA_DIR)}")
    print("🌐 Server will be available at: http://localhost:8000")
    print("📊 Health check: http://localhost:8000/health")
    print("📋 List files: http://localhost:8000/list_annotation_files")
    print("\n" + "="*50)
    
    app.run(debug=True, host='0.0.0.0', port=8000)
