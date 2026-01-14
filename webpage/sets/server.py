#!/usr/bin/env python3
"""
Flask server for diagnosis annotation tool.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
from datetime import datetime

from flask import redirect
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request

from werkzeug.middleware.proxy_fix import ProxyFix


app = Flask(__name__)
CORS(app)

app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)

CLIENT_SECRET_JSON = os.getenv("GOOGLE_CLIENT_SECRET_JSON")
TOKEN_JSON = os.getenv("GOOGLE_TOKEN_JSON")
SCOPES = os.getenv("SCOPES", "https://www.googleapis.com/auth/drive.file").split()
REDIRECT_URI = os.getenv("REDIRECT_URI")
FOLDER_ID = os.getenv("FOLDER_ID")

# Configuration
SETS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SETS_DIR, "annotation_manager", "data")
DATA_PATH = os.environ.get("DATA_PATH",os.path.join(DATA_DIR, "sampled_data_HCM-3k.json"),)
ANNOTATIONS_PATH = os.environ.get("ANNOTATIONS_PATH",os.path.join(DATA_DIR, "annotations.json"),)

questions_data = None
annotations = {}

APP_VERSION = "v2-show-token-page-2026-01-14"
app.logger.warning("BOOT %s file=%s", APP_VERSION, __file__)

@app.route("/version")
def version():
    return jsonify({"version": APP_VERSION, "file": __file__})

@app.route('/authorize')
def authorize():
    if not CLIENT_SECRET_JSON:
        return jsonify({'error': 'Missing GOOGLE_CLIENT_SECRET_JSON'}), 400

    if not REDIRECT_URI:
        return jsonify({'error': 'Missing REDIRECT_URI'}), 400

    flow = Flow.from_client_config(
        json.loads(CLIENT_SECRET_JSON),
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    auth_url, _ = flow.authorization_url(prompt='consent')
    return redirect(auth_url)

@app.route('/oauth2callback')
def oauth2callback():
    if not CLIENT_SECRET_JSON:
        return "❌ Missing client secret.", 400

    flow = Flow.from_client_config(
        json.loads(CLIENT_SECRET_JSON),
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    flow.fetch_token(authorization_response=request.url)
    creds = flow.credentials

    token_json = creds.to_json()

    # Still print for logs (optional)
    app.logger.warning("OAUTH TOKEN JSON: %s", token_json)

    # Show it on the page so you can copy it
    return f"""
    <h2>✅ Authentication successful</h2>
    <p>Copy the token JSON below into Render env var <b>GOOGLE_TOKEN_JSON</b>, then redeploy.</p>
    <pre style="white-space: pre-wrap; word-break: break-word; padding: 12px; background: #f5f5f5; border: 1px solid #ddd;">
{token_json}
    </pre>
    """


def get_drive_service():
    token_json = os.getenv("GOOGLE_TOKEN_JSON")
    if not token_json:
        raise Exception("❌ No token found. Run /authorize first.")

    creds = Credentials.from_authorized_user_info(json.loads(token_json), SCOPES)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        print("🔄 Access token refreshed successfully.", flush=True)

        # Optional but recommended: print the updated token so you can copy it back into Render env
        print("🔁 Updated token JSON (copy to GOOGLE_TOKEN_JSON):", flush=True)
        print(creds.to_json(), flush=True)

    return build('drive', 'v3', credentials=creds)


def upload_annotations_to_drive(local_path: str):
    """
    Upload/update annotations.json in the Drive folder FOLDER_ID.
    Uses the same pattern as your CSV uploader: update if exists, else create.
    """
    if not FOLDER_ID:
        raise Exception("❌ Missing FOLDER_ID env var")

    service = get_drive_service()
    file_name = os.path.basename(local_path)
    query = f"name='{file_name}' and '{FOLDER_ID}' in parents and trashed=false"

    existing_files = service.files().list(q=query, fields="files(id, name)").execute().get('files', [])
    media = MediaFileUpload(local_path, mimetype="application/json", resumable=True)

    if existing_files:
        file_id = existing_files[0]['id']
        updated_file = service.files().update(
            fileId=file_id,
            media_body=media,
            fields='id, name, webViewLink'
        ).execute()
        print(f"🔄 Updated Drive annotations file: {updated_file.get('webViewLink')}")
        return updated_file.get("webViewLink")
    else:
        file_metadata = {'name': file_name, 'parents': [FOLDER_ID]}
        new_file = service.files().create(
            body=file_metadata,
            media_body=media,
            fields='id, name, webViewLink'
        ).execute()
        print(f"✅ Uploaded new Drive annotations file: {new_file.get('webViewLink')}")
        return new_file.get("webViewLink")

def load_data():
    """Load questions data from JSON file."""
    global questions_data
    if questions_data is None:
        with open(DATA_PATH, 'r') as f:
            data = json.load(f)
            questions_data = data['per_sample']
    return questions_data

def load_annotations():
    """Load existing annotations."""
    global annotations
    if os.path.exists(ANNOTATIONS_PATH):
        with open(ANNOTATIONS_PATH, 'r') as f:
            annotations = json.load(f)
    else:
        annotations = {}
    return annotations

def save_annotations(annotations_dict):
    os.makedirs(os.path.dirname(ANNOTATIONS_PATH), exist_ok=True)
    with open(ANNOTATIONS_PATH, 'w') as f:
        json.dump(annotations_dict, f, indent=2)

    try:
        link = upload_annotations_to_drive(ANNOTATIONS_PATH)
        return link
    except Exception as e:
        print(f"⚠️ Google Drive upload skipped due to error: {e}")
        return None


def get_next_unannotated_question():
    """Get the next unannotated question."""
    questions = load_data()
    annotations = load_annotations()
    
    for i, question in enumerate(questions):
        if str(question['index']) not in annotations:
            return question, i
    return None, None

def get_stats():
    """Get annotation statistics."""
    questions = load_data()
    annotations = load_annotations()
    total = len(questions)
    annotated = len(annotations)
    return {
        'total': total,
        'annotated': annotated,
        'remaining': total - annotated
    }

@app.route('/')
def serve_index():
    return send_from_directory(SETS_DIR, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(SETS_DIR, filename)

@app.route('/get_next_question', methods=['GET'])
def get_next_question():
    """Get the next unannotated question."""
    try:
        question, idx = get_next_unannotated_question()
        
        if question is None:
            stats = get_stats()
            return jsonify({
                'success': True,
                'completed': True,
                'stats': stats
            })
        
        stats = get_stats()
        return jsonify({
            'success': True,
            'completed': False,
            'question': {
                'index': question['index'],
                'input': question['input'],
                'plausible_set': question['judge_dx_space']['plausible_set'],
                'highly_likely_set': question['judge_dx_space']['highly_likely_set']
            },
            'stats': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    """Save annotation and get next question."""
    try:
        data = request.get_json()
        
        if not data or 'index' not in data:
            return jsonify({'error': 'No index provided'}), 400
        
        question_index = str(data['index'])
        annotation = {
            'index': question_index,
            'not_plausible': data.get('not_plausible', []),
            'missing_plausible': data.get('missing_plausible', ''),
            'not_highly_likely': data.get('not_highly_likely', []),
            'missing_highly_likely': data.get('missing_highly_likely', ''),
            'timestamp': datetime.now().isoformat()
        }
        
        annotations_dict = load_annotations()
        annotations_dict[question_index] = annotation
        drive_link = save_annotations(annotations_dict)
        
        # Get next question
        question, idx = get_next_unannotated_question()
        
        if question is None:
            stats = get_stats()
            return jsonify({
                'success': True,
                'completed': True,
                'stats': stats
            })
        
        stats = get_stats()
        return jsonify({
            'success': True,
            'completed': False,
            'question': {
                'index': question['index'],
                'input': question['input'],
                'plausible_set': question['judge_dx_space']['plausible_set'],
                'highly_likely_set': question['judge_dx_space']['highly_likely_set']
            },
            'stats': stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'data_path': DATA_PATH,
        'data_exists': os.path.exists(DATA_PATH),
        'annotations_path': ANNOTATIONS_PATH
    })

if __name__ == '__main__':
    print("🚀 Starting Diagnosis Annotation Server...")
    print(f"📊 Data file: {DATA_PATH}")
    print(f"💾 Annotations file: {ANNOTATIONS_PATH}")
    print("🌐 Server will be available at: http://localhost:5002")
    print("\n" + "="*50)

    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port)

