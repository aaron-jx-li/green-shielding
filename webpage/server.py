#!/usr/bin/env python3
"""
Flask server for handling annotation saves.
This server receives annotation data from the web interface and saves it to local files.
"""

from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS
import os, io, json
import pandas as pd
from datetime import datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
import traceback


app = Flask(__name__)
CORS(app)

CLIENT_SECRET_JSON = os.getenv("GOOGLE_CLIENT_SECRET_JSON")
TOKEN_JSON = os.getenv("GOOGLE_TOKEN_JSON")
SCOPES = os.getenv("SCOPES", "https://www.googleapis.com/auth/drive.file").split()
REDIRECT_URI = os.getenv(
    "REDIRECT_URI",
    "http://localhost:8000/oauth2callback"
)
FOLDER_ID = os.getenv("FOLDER_ID")
DATA_DIR = os.getenv("DATA_DIR", "annotations")
os.makedirs(DATA_DIR, exist_ok=True)

@app.route('/authorize')
def authorize():
    """Generate OAuth authorization URL."""
    if not CLIENT_SECRET_JSON:
        return jsonify({'error': 'Missing GOOGLE_CLIENT_SECRET_JSON'}), 400

    flow = Flow.from_client_config(
        json.loads(CLIENT_SECRET_JSON),
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    auth_url, _ = flow.authorization_url(prompt='consent')
    return redirect(auth_url)



@app.route('/oauth2callback')
def oauth2callback():
    """OAuth callback: exchange code for tokens."""
    if not CLIENT_SECRET_JSON:
        return "❌ Missing client secret.", 400

    flow = Flow.from_client_config(
        json.loads(CLIENT_SECRET_JSON),
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    flow.fetch_token(authorization_response=request.url)
    creds = flow.credentials

    print("✅ OAuth complete! Copy this token and add it to Render as GOOGLE_TOKEN_JSON:\n")
    print(creds.to_json())

    return (
        "✅ Authentication successful! Check your Render logs and copy the token JSON into your Render environment as GOOGLE_TOKEN_JSON."
    )


def get_drive_service():
    """Build Drive service using env-stored token."""
    creds = None
    if TOKEN_JSON:
        creds = Credentials.from_authorized_user_info(json.loads(TOKEN_JSON), SCOPES)

    if not creds or not creds.valid:
        raise Exception("❌ Not authenticated. Visit /authorize first.")
    return build('drive', 'v3', credentials=creds)


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
    try:
        annotation_data = request.get_json()
        if not annotation_data:
            return jsonify({'error': 'No data received'}), 400

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = f"{DATA_DIR}/annotations_{timestamp}.json"
        csv_filename = f"{DATA_DIR}/annotations_{timestamp}.csv"

        # Save locally
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, indent=2, ensure_ascii=False)
        df = pd.DataFrame(annotation_data)
        df.to_csv(csv_filename, index=False, encoding='utf-8')

        # Save latest
        latest_json = f"{DATA_DIR}/latest_annotations.json"
        latest_csv = f"{DATA_DIR}/latest_annotations.csv"
        with open(latest_json, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, indent=2, ensure_ascii=False)
        df.to_csv(latest_csv, index=False, encoding='utf-8')

        print(f"✅ Saved annotations to local files.")

        # Upload to Drive
        service = get_drive_service()
        FOLDER_ID = "1uSsq_UbH_BYu_SB_7Xh1dJaaxesgb9O4"
        uploaded_files = []

        for file_path in [json_filename, csv_filename, latest_json, latest_csv]:
            file_metadata = {
                'name': os.path.basename(file_path),
                'parents': [FOLDER_ID]
            }
            media = MediaFileUpload(file_path, resumable=True)
            file = service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id, name, webViewLink'
            ).execute()
            uploaded_files.append({
                'name': file['name'],
                'id': file['id'],
                'link': file['webViewLink']
            })

        print("✅ Uploaded files to Google Drive folder:")
        for f in uploaded_files:
            print(f"   - {f['name']} → {f['link']}")

        return jsonify({
            'success': True,
            'message': 'Annotations saved and uploaded to Drive!',
            'files_created': [json_filename, csv_filename, latest_json, latest_csv],
            'drive_uploads': uploaded_files,
            'timestamp': timestamp
        })

    except Exception as e:
        print(f"❌ Error saving annotations: {e}")
        traceback.print_exc()
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
    port = int(os.environ.get("PORT", 10000))
    print("🚀 Starting Annotation Server...")
    print(f"📁 Annotations will be saved to: {os.path.abspath(DATA_DIR)}")
    print(f"🌐 Server will be available at: 0.0.0.0:{port}")
    print("📊 Health check: /health")
    print("📋 List files: /list_annotation_files")
    print("\n" + "="*50)

    app.run(host='0.0.0.0', port=port, debug=False)