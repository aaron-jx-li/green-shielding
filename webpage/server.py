#!/usr/bin/env python3
"""
Flask annotation server with Google Drive upload.
Runs safely on Render using environment variables for credentials.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os, io, json
import pandas as pd
from datetime import datetime
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

app = Flask(__name__)
CORS(app)

# === Environment Configuration ===
CLIENT_SECRET_JSON = os.getenv("GOOGLE_CLIENT_SECRET_JSON")
TOKEN_JSON = os.getenv("GOOGLE_TOKEN_JSON")  # stored token (optional)
SCOPES = os.getenv("SCOPES", "https://www.googleapis.com/auth/drive.file").split()
REDIRECT_URI = os.getenv("REDIRECT_URI", "http://localhost:8000/oauth2callback")
FOLDER_ID = os.getenv("FOLDER_ID")
DATA_DIR = os.getenv("DATA_DIR", "annotations")
os.makedirs(DATA_DIR, exist_ok=True)


# === Google Auth ===
def get_drive_service():
    """Build Drive service using env-stored token."""
    creds = None
    if TOKEN_JSON:
        creds = Credentials.from_authorized_user_info(json.loads(TOKEN_JSON), SCOPES)

    if not creds or not creds.valid:
        raise Exception("❌ Not authenticated. Visit /authorize first.")
    return build('drive', 'v3', credentials=creds)


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
    return jsonify({'auth_url': auth_url})


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


# === Annotation saving ===
@app.route('/save_annotations', methods=['POST'])
def save_annotations():
    """Save annotation data locally and upload to Google Drive."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data received'}), 400

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(DATA_DIR, f"annotations_{timestamp}.json")
        csv_path = os.path.join(DATA_DIR, f"annotations_{timestamp}.csv")

        # Save local files
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        pd.DataFrame(data).to_csv(csv_path, index=False, encoding="utf-8")

        # Upload to Drive
        service = get_drive_service()
        uploaded = []

        for file_path in [json_path, csv_path]:
            file_metadata = {"name": os.path.basename(file_path)}
            if FOLDER_ID:
                file_metadata["parents"] = [FOLDER_ID]

            media = MediaFileUpload(file_path, resumable=True)
            file = (
                service.files()
                .create(body=file_metadata, media_body=media, fields="id, name, webViewLink")
                .execute()
            )
            uploaded.append(file)

        return jsonify({
            "success": True,
            "files_uploaded": uploaded,
            "timestamp": timestamp
        })

    except Exception as e:
        print(f"❌ Error saving annotations: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/health')
def health():
    return jsonify({'status': 'ok'})


if __name__ == '__main__':
    port = int(os.getenv("PORT", 8000))
    print("🚀 Annotation server running on port", port)
    app.run(host='0.0.0.0', port=port, debug=False)
