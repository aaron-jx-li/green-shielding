#!/usr/bin/env python3
"""
Flask server for diagnosis annotation tool.
Multi-user version with per-question assignments + per-user progress,
while keeping Google OAuth + Drive upload.
"""

from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS
import json
import os
import random
from datetime import datetime

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

# =========================
# Configuration / Paths
# =========================
SETS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SETS_DIR, "annotation_manager", "data")

DATA_PATH = os.environ.get(
    "DATA_PATH",
    os.path.join(DATA_DIR, "sampled_data_HCM-3k.json"),
)
ANNOTATIONS_PATH = os.environ.get(
    "ANNOTATIONS_PATH",
    os.path.join(DATA_DIR, "annotations.json"),
)
ASSIGNMENTS_PATH = os.environ.get(
    "ASSIGNMENTS_PATH",
    os.path.join(DATA_DIR, "user_assignments.json"),
)

# =========================
# Multi-user configuration
# =========================
USERS = ["Dr. Kornblith", "Dr. Bains"]
USERS_PER_QUESTION = 2  # number of users assigned to each question

# =========================
# Globals
# =========================
questions_data = None
annotations = {}

APP_VERSION = "v2-multiuser-assignments-2026-01-21"
app.logger.warning("BOOT %s file=%s", APP_VERSION, __file__)


@app.route("/version")
def version():
    return jsonify({"version": APP_VERSION, "file": __file__})


# =========================
# OAuth / Drive upload
# =========================
@app.route("/authorize")
def authorize():
    if not CLIENT_SECRET_JSON:
        return jsonify({"error": "Missing GOOGLE_CLIENT_SECRET_JSON"}), 400
    if not REDIRECT_URI:
        return jsonify({"error": "Missing REDIRECT_URI"}), 400

    flow = Flow.from_client_config(
        json.loads(CLIENT_SECRET_JSON),
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
    )
    auth_url, _ = flow.authorization_url(prompt="consent")
    return redirect(auth_url)


@app.route("/oauth2callback")
def oauth2callback():
    if not CLIENT_SECRET_JSON:
        return "❌ Missing client secret.", 400

    flow = Flow.from_client_config(
        json.loads(CLIENT_SECRET_JSON),
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
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

        # Optional: print updated token for Render env refresh
        print("🔁 Updated token JSON (copy to GOOGLE_TOKEN_JSON):", flush=True)
        print(creds.to_json(), flush=True)

    return build("drive", "v3", credentials=creds)


def upload_annotations_to_drive(local_path: str):
    """
    Upload/update annotations.json in the Drive folder FOLDER_ID.
    Update if exists, else create.
    """
    if not FOLDER_ID:
        raise Exception("❌ Missing FOLDER_ID env var")

    service = get_drive_service()
    file_name = os.path.basename(local_path)
    query = f"name='{file_name}' and '{FOLDER_ID}' in parents and trashed=false"

    existing_files = (
        service.files().list(q=query, fields="files(id, name)").execute().get("files", [])
    )
    media = MediaFileUpload(local_path, mimetype="application/json", resumable=True)

    if existing_files:
        file_id = existing_files[0]["id"]
        updated_file = (
            service.files()
            .update(fileId=file_id, media_body=media, fields="id, name, webViewLink")
            .execute()
        )
        print(f"🔄 Updated Drive annotations file: {updated_file.get('webViewLink')}")
        return updated_file.get("webViewLink")
    else:
        file_metadata = {"name": file_name, "parents": [FOLDER_ID]}
        new_file = (
            service.files()
            .create(body=file_metadata, media_body=media, fields="id, name, webViewLink")
            .execute()
        )
        print(f"✅ Uploaded new Drive annotations file: {new_file.get('webViewLink')}")
        return new_file.get("webViewLink")


# =========================
# Data / Assignments / Annotations
# =========================
def load_data():
    """Load questions data from JSON file."""
    global questions_data
    if questions_data is None:
        with open(DATA_PATH, "r") as f:
            data = json.load(f)
            # script #2 uses data['per_sample'] -> each question has question['index'], etc.
            questions_data = data["per_sample"]
    return questions_data


def load_annotations():
    """Load existing annotations (nested per-user)."""
    global annotations
    if os.path.exists(ANNOTATIONS_PATH):
        with open(ANNOTATIONS_PATH, "r") as f:
            annotations = json.load(f)
    else:
        annotations = {}
    return annotations


def save_annotations(annotations_dict):
    """Save annotations to JSON + upload to Drive."""
    os.makedirs(os.path.dirname(ANNOTATIONS_PATH), exist_ok=True)
    with open(ANNOTATIONS_PATH, "w") as f:
        json.dump(annotations_dict, f, indent=2)

    try:
        link = upload_annotations_to_drive(ANNOTATIONS_PATH)
        return link
    except Exception as e:
        print(f"⚠️ Google Drive upload skipped due to error: {e}")
        return None


def load_assignments():
    """Load question-to-users assignments."""
    if os.path.exists(ASSIGNMENTS_PATH):
        with open(ASSIGNMENTS_PATH, "r") as f:
            return json.load(f)
    return {}


def save_assignments(assignments_dict):
    """Save question-to-users assignments."""
    os.makedirs(os.path.dirname(ASSIGNMENTS_PATH), exist_ok=True)
    with open(ASSIGNMENTS_PATH, "w") as f:
        json.dump(assignments_dict, f, indent=2)


def initialize_assignments():
    """
    Initialize question assignments by randomly assigning each question to N users.
    Idempotent: only assigns previously-unseen question indices.
    """
    questions = load_data()
    assignments = load_assignments()

    question_indices = [str(q["__idx"]) for q in questions]
    existing_indices = set(assignments.keys())
    new_indices = set(question_indices) - existing_indices

    if not new_indices:
        return assignments

    for q_idx in new_indices:
        num_users = min(USERS_PER_QUESTION, len(USERS))
        assignments[q_idx] = random.sample(USERS, num_users)

    save_assignments(assignments)
    return assignments


def get_user_next_question(user_id: str):
    """Get the next unannotated question for a specific user (assigned-to-user only)."""
    questions = load_data()
    annotations_dict = load_annotations()
    assignments = load_assignments()

    if not assignments:
        assignments = initialize_assignments()

    for question in questions:
        q_idx = str(question["__idx"])
        if q_idx in assignments and user_id in assignments[q_idx]:
            if q_idx not in annotations_dict or user_id not in annotations_dict.get(q_idx, {}):
                return question
    return None


def get_user_stats(user_id: str):
    """Get annotation statistics for a specific user (assigned vs done)."""
    questions = load_data()
    annotations_dict = load_annotations()
    assignments = load_assignments()

    if not assignments:
        assignments = initialize_assignments()

    total = 0
    annotated = 0

    for question in questions:
        q_idx = str(question["__idx"])
        if q_idx in assignments and user_id in assignments[q_idx]:
            total += 1
            if q_idx in annotations_dict and user_id in annotations_dict[q_idx]:
                annotated += 1

    return {"total": total, "annotated": annotated, "remaining": total - annotated}


# =========================
# Static serving
# =========================
@app.route("/")
def serve_index():
    return send_from_directory(SETS_DIR, "index.html")


@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(SETS_DIR, filename)


# =========================
# Multi-user endpoints
# =========================
@app.route("/users", methods=["GET"])
def get_users():
    return jsonify({"success": True, "users": USERS})


@app.route("/login", methods=["POST"])
def login():
    """Validate user and ensure assignments exist."""
    try:
        data = request.get_json() or {}
        user_id = data.get("user_id")

        if not user_id:
            return jsonify({"success": False, "error": "No user_id provided"}), 400
        if user_id not in USERS:
            return jsonify({"success": False, "error": f"Invalid user_id. Must be one of: {USERS}"}), 400

        initialize_assignments()

        return jsonify({"success": True, "user_id": user_id, "users": USERS})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/stats", methods=["GET"])
def stats():
    """Get user-specific stats."""
    try:
        user_id = request.args.get("user_id")
        if not user_id:
            return jsonify({"success": False, "error": "user_id parameter required"}), 400
        if user_id not in USERS:
            return jsonify({"success": False, "error": f"Invalid user_id. Must be one of: {USERS}"}), 400

        s = get_user_stats(user_id)
        return jsonify({"success": True, "stats": s})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =========================
# Core API (now per-user)
# =========================
@app.route("/get_next_question", methods=["GET"])
def get_next_question():
    """Get the next unannotated question for a user."""
    try:
        user_id = request.args.get("user_id")
        if not user_id:
            return jsonify({"success": False, "error": "user_id parameter required"}), 400
        if user_id not in USERS:
            return jsonify({"success": False, "error": f"Invalid user_id. Must be one of: {USERS}"}), 400

        question = get_user_next_question(user_id)

        if question is None:
            return jsonify(
                {"success": True, "completed": True, "stats": get_user_stats(user_id)}
            )

        gt = question["ground_truth_space_majority"]
        return jsonify(
            {
                "success": True,
                "completed": False,
                "question": {
                    "index": question["__idx"],
                    "input": question["raw_input"],
                    "plausible_set": gt["plausible_set"],
                    "highly_likely_set": gt["highly_likely_set"],
                    "cannot_miss_set": gt["cannot_miss_set"],
                },
                "stats": get_user_stats(user_id),
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/save_annotation", methods=["POST"])
def save_annotation():
    """Save annotation for a specific user and return next question for that user."""
    try:
        data = request.get_json() or {}

        # Match reference server requirements
        if "__idx" not in data:
            return jsonify({"success": False, "error": "No index provided"}), 400
        if "user_id" not in data:
            return jsonify({"success": False, "error": "No user_id provided"}), 400

        user_id = data["user_id"]
        if user_id not in USERS:
            return jsonify({"success": False, "error": f"Invalid user_id. Must be one of: {USERS}"}), 400

        question_index = str(data["__idx"])

        # Make sure assignments exist and user is actually assigned this question
        assignments = load_assignments()
        if not assignments:
            assignments = initialize_assignments()

        if question_index not in assignments or user_id not in assignments[question_index]:
            return jsonify(
                {
                    "success": False,
                    "error": "User is not assigned to this question (or assignments not initialized).",
                }
            ), 403

        # Save annotation under nested structure {question_index: {user_id: annotation}}
        annotation = {
            "not_plausible": data.get("not_plausible", []),
            "missing_plausible": data.get("missing_plausible", ""),
            "not_highly_likely": data.get("not_highly_likely", []),
            "missing_highly_likely": data.get("missing_highly_likely", ""),
            "not_cannot_miss": data.get("not_cannot_miss", []),
            "missing_cannot_miss": data.get("missing_cannot_miss", ""),
            "timestamp": datetime.now().isoformat(),
        }

        annotations_dict = load_annotations()
        if question_index not in annotations_dict or not isinstance(annotations_dict.get(question_index), dict):
            annotations_dict[question_index] = {}
        annotations_dict[question_index][user_id] = annotation

        save_annotations(annotations_dict)  # (your version may also upload to Drive)

        # Return next question for this user
        next_q = get_user_next_question(user_id)
        if next_q is None:
            return jsonify(
                {"success": True, "completed": True, "stats": get_user_stats(user_id)}
            )

        gt_next = next_q["ground_truth_space_majority"]
        return jsonify(
            {
                "success": True,
                "completed": False,
                "question": {
                    "index": next_q["__idx"],
                    "input": next_q["raw_input"],
                    "plausible_set": gt_next["plausible_set"],
                    "highly_likely_set": gt_next["highly_likely_set"],
                    "cannot_miss_set": gt_next["cannot_miss_set"],
                },
                "stats": get_user_stats(user_id),
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/initialize_assignments", methods=["POST"])
def initialize_assignments_endpoint():
    """Initialize question assignments (idempotent)."""
    try:
        assignments = initialize_assignments()
        return jsonify(
            {
                "success": True,
                "message": "Assignments initialized",
                "total_questions": len(assignments),
            }
        )
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify(
        {
            "status": "healthy",
            "data_path": DATA_PATH,
            "data_exists": os.path.exists(DATA_PATH),
            "annotations_path": ANNOTATIONS_PATH,
            "assignments_path": ASSIGNMENTS_PATH,
            "annotations_exists": os.path.exists(ANNOTATIONS_PATH),
            "assignments_exists": os.path.exists(ASSIGNMENTS_PATH),
        }
    )


if __name__ == "__main__":
    print("🚀 Starting Diagnosis Annotation Server...")
    print(f"📊 Data file: {DATA_PATH}")
    print(f"💾 Annotations file: {ANNOTATIONS_PATH}")
    print(f"👥 Assignments file: {ASSIGNMENTS_PATH}")
    print("🌐 Server will be available at: http://localhost:5002")
    print("\n" + "=" * 50)

    port = int(os.environ.get("PORT", 5002))
    app.run(host="0.0.0.0", port=port)
