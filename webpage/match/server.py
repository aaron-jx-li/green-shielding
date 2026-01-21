#!/usr/bin/env python3
"""
Flask server for dynamic annotation with server-side question selection.
Render-friendly + Google Drive upload + CSV lock + multi-user assignments.

This file INCORPORATES the old start_server.py logic:
- default local DATA_CONFIG
- path validation + helpful logs
- configure(...) function (still supported)
- env var override (Render best practice)
"""

from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS
import json
import os
import time
from contextlib import contextmanager
import pandas as pd
import csv

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request

from werkzeug.middleware.proxy_fix import ProxyFix

# -----------------------------------------------------------------------------
# App setup
# -----------------------------------------------------------------------------
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
CORS(app)

# -----------------------------------------------------------------------------
# Google OAuth / Drive env config (Render Dashboard -> Environment)
# -----------------------------------------------------------------------------
CLIENT_SECRET_JSON = os.getenv("GOOGLE_CLIENT_SECRET_JSON")
TOKEN_JSON = os.getenv("GOOGLE_TOKEN_JSON")
SCOPES = os.getenv("SCOPES", "https://www.googleapis.com/auth/drive.file").split()
REDIRECT_URI = os.getenv("REDIRECT_URI")
FOLDER_ID = os.getenv("FOLDER_ID")

# -----------------------------------------------------------------------------
# Local default configuration (from your start_server.py)
# These are used when env vars are not provided (local dev).
# -----------------------------------------------------------------------------
DATA_CONFIG = {
    "all_questions_metadata_csv_path": "./annotation_manager/ak_review_round0/all_questions_metadata.csv",
    "all_questions_json_path": "./annotation_manager/ak_review_round0/all_questions.json",
    "slrt_bounds_csv_path": "./annotation_manager/ak_review_round0/slrt_test_low_upper_bound.csv",
    "expert_dec_column": "expert_dec",
    "unannotated_value": -1,
    "match_value": 3,
    "close_match_value": 2,
    "vague_match_value": 1,
    "no_match_value": 0,
}

# -----------------------------------------------------------------------------
# Global configuration (the server reads from CONFIG)
# IMPORTANT: env vars override defaults (Render).
# -----------------------------------------------------------------------------
CONFIG = {
    # data paths (env overrides default)
    "all_questions_metadata_csv_path": os.getenv("ALL_QUESTIONS_METADATA_CSV_PATH", DATA_CONFIG["all_questions_metadata_csv_path"]),
    "all_questions_json_path": os.getenv("ALL_QUESTIONS_JSON_PATH", DATA_CONFIG["all_questions_json_path"]),
    "slrt_bounds_csv_path": os.getenv("SLRT_BOUNDS_CSV_PATH", DATA_CONFIG["slrt_bounds_csv_path"]),

    # annotation values (env overrides default)
    "expert_dec_column": os.getenv("EXPERT_DEC_COLUMN", DATA_CONFIG["expert_dec_column"]),
    "unannotated_value": int(os.getenv("UNANNOTATED_VALUE", str(DATA_CONFIG["unannotated_value"]))),
    "match_value": int(os.getenv("MATCH_VALUE", str(DATA_CONFIG["match_value"]))),
    "close_match_value": int(os.getenv("CLOSE_MATCH_VALUE", str(DATA_CONFIG["close_match_value"]))),
    "vague_match_value": int(os.getenv("VAGUE_MATCH_VALUE", str(DATA_CONFIG["vague_match_value"]))),
    "no_match_value": int(os.getenv("NO_MATCH_VALUE", str(DATA_CONFIG["no_match_value"]))),
}

# Global user assignment tracking: user_id -> set of csv_indices
user_assignments = {}

# -----------------------------------------------------------------------------
# Startup/config helpers (incorporated from start_server.py)
# -----------------------------------------------------------------------------
def _abs_path(p: str) -> str:
    """Convert relative path to absolute (relative to this file)."""
    if not p:
        return p
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(os.path.dirname(__file__), p))


def validate_data_paths():
    """Validate that configured data files exist. Returns (ok, csv_path, json_path, slrt_path, error_msg)."""
    csv_path = _abs_path(CONFIG["all_questions_metadata_csv_path"])
    json_path = _abs_path(CONFIG["all_questions_json_path"])
    slrt_path = _abs_path(CONFIG["slrt_bounds_csv_path"])

    if not os.path.exists(json_path):
        return False, None, None, None, f"All questions JSON file not found: {json_path}"

    if not os.path.exists(csv_path):
        return False, None, None, None, f"All questions metadata CSV file not found: {csv_path}"

    if not os.path.exists(slrt_path):
        return False, None, None, None, f"SLRT bounds CSV file not found: {slrt_path}"

    # store abs paths back into CONFIG so all later reads are consistent
    CONFIG["all_questions_metadata_csv_path"] = csv_path
    CONFIG["all_questions_json_path"] = json_path
    CONFIG["slrt_bounds_csv_path"] = slrt_path

    return True, csv_path, json_path, slrt_path, None


def validate_config():
    """Validate that the CSV/JSON files exist (used by /health)."""
    ok, _, _, _, err = validate_data_paths()
    return ok, err


def print_startup_banner():
    print("=" * 60, flush=True)
    print("🏥 Dynamic Annotation Tool - Server Startup", flush=True)
    print("=" * 60, flush=True)

    print("\n📋 Configuration (effective):", flush=True)
    print(f"   All questions CSV:  {CONFIG['all_questions_metadata_csv_path']}", flush=True)
    print(f"   All questions JSON: {CONFIG['all_questions_json_path']}", flush=True)
    print(f"   SLRT bounds CSV:    {CONFIG['slrt_bounds_csv_path']}", flush=True)
    print(f"   Column:            {CONFIG['expert_dec_column']}", flush=True)
    print(
        f"   Values: Match={CONFIG['match_value']}, Close={CONFIG['close_match_value']}, "
        f"Vague={CONFIG['vague_match_value']}, No={CONFIG['no_match_value']}",
        flush=True,
    )
    print("=" * 60, flush=True)


def configure(all_questions_metadata_csv_path, all_questions_json_path, slrt_bounds_csv_path,
              expert_dec_column="expert_dec",
              unannotated_value=-1, match_value=3, close_match_value=2, vague_match_value=1, no_match_value=0):
    """
    Backwards-compatible configure() like your old server.
    You can still call this from a script, but on Render you should rely on env vars.
    """
    CONFIG["all_questions_metadata_csv_path"] = all_questions_metadata_csv_path
    CONFIG["all_questions_json_path"] = all_questions_json_path
    CONFIG["slrt_bounds_csv_path"] = slrt_bounds_csv_path
    CONFIG["expert_dec_column"] = expert_dec_column
    CONFIG["unannotated_value"] = unannotated_value
    CONFIG["match_value"] = match_value
    CONFIG["close_match_value"] = close_match_value
    CONFIG["vague_match_value"] = vague_match_value
    CONFIG["no_match_value"] = no_match_value

    ok, _, _, _, err = validate_data_paths()
    if not ok:
        print(f"❌ Configuration error: {err}", flush=True)
        return False

    stats = get_stats()
    if stats:
        print(f"✅ Loaded CSV with {stats['total']} total questions", flush=True)
        print(f"   Unannotated: {stats['remaining']}", flush=True)
        print(f"   Annotated:   {stats['annotated']}", flush=True)

    return True


# -----------------------------------------------------------------------------
# Google auth routes (guideline-style)
# -----------------------------------------------------------------------------
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
    if not REDIRECT_URI:
        return "❌ Missing REDIRECT_URI.", 400

    flow = Flow.from_client_config(
        json.loads(CLIENT_SECRET_JSON),
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI,
    )
    flow.fetch_token(authorization_response=request.url)
    creds = flow.credentials

    print("✅ OAuth complete! Copy this token JSON and set GOOGLE_TOKEN_JSON in Render:\n", flush=True)
    print(creds.to_json(), flush=True)

    return "✅ Authentication successful! Check Render logs and copy token JSON into GOOGLE_TOKEN_JSON."


def get_drive_service():
    if not TOKEN_JSON:
        raise Exception("❌ No token found. Run /authorize first.")
    creds = Credentials.from_authorized_user_info(json.loads(TOKEN_JSON), SCOPES)
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        print("🔄 Access token refreshed successfully.", flush=True)
    return build("drive", "v3", credentials=creds)


# -----------------------------------------------------------------------------
# CSV file lock (your multi-user-safe version)
# -----------------------------------------------------------------------------
def get_lock_file_path():
    csv_path = CONFIG["all_questions_metadata_csv_path"]
    if not csv_path:
        return None
    return csv_path + ".lock"


def acquire_csv_lock(timeout=10):
    lock_path = get_lock_file_path()
    if not lock_path:
        return False

    start_time = time.time()
    wait_time = 0.1
    while time.time() - start_time < timeout:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return True
        except FileExistsError:
            time.sleep(wait_time)
            wait_time = min(wait_time * 1.5, 0.5)
        except Exception as e:
            print(f"❌ Error acquiring lock: {e}", flush=True)
            return False

    print(f"❌ Timeout acquiring lock after {timeout} seconds", flush=True)
    return False


def release_csv_lock():
    lock_path = get_lock_file_path()
    if not lock_path:
        return
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except Exception as e:
        print(f"❌ Error releasing lock: {e}", flush=True)


@contextmanager
def with_csv_lock():
    if not acquire_csv_lock():
        raise Exception("Failed to acquire CSV lock")
    try:
        yield
    finally:
        release_csv_lock()


# -----------------------------------------------------------------------------
# Core logic
# -----------------------------------------------------------------------------
def get_slrt_random_unannotated_question(user_id: str):
    """Pick a random unannotated question for this user (not already assigned)."""
    with with_csv_lock():
        if user_id not in user_assignments:
            user_assignments[user_id] = set()
        user_assigned_indices = user_assignments[user_id]

        questions_df = pd.read_csv(CONFIG["all_questions_metadata_csv_path"], quoting=csv.QUOTE_ALL)
        slrt_bounds_df = pd.read_csv(CONFIG["slrt_bounds_csv_path"], quoting=csv.QUOTE_ALL)
        with open(CONFIG["all_questions_json_path"], "r") as f:
            questions_json = json.load(f)

        poss_cats = questions_df["category"].unique()
        for curr_cat in poss_cats:
            cat_questions_df = questions_df[questions_df["category"] == curr_cat]
            cat_num_seen = (~cat_questions_df["to_be_seen"].astype(bool)).sum()

            min_samples = slrt_bounds_df.n.min()
            max_samples = slrt_bounds_df.n.max()
            if cat_num_seen < min_samples or cat_num_seen > max_samples:
                continue

            row = slrt_bounds_df[slrt_bounds_df["n"] == cat_num_seen]
            if row.empty:
                continue

            cat_low_bound = row["lower"].iloc[0]
            cat_high_bound = row["upper"].iloc[0]
            num_incorrect = cat_questions_df[cat_questions_df[CONFIG["expert_dec_column"]].isin([0, 1])].shape[0]

            if (num_incorrect >= cat_high_bound) or (num_incorrect <= cat_low_bound):
                questions_df.loc[questions_df["category"] == curr_cat, "to_be_seen"] = False

        # Persist updated to_be_seen
        questions_df.to_csv(CONFIG["all_questions_metadata_csv_path"], index=False, quoting=csv.QUOTE_ALL)

        # Filter: still to be seen + not assigned to this user
        available_questions = questions_df[
            (questions_df["to_be_seen"] == True) &
            (~questions_df["Index"].isin(user_assigned_indices))
        ]

        if len(available_questions) == 0:
            print(f"No more questions available for user {user_id}", flush=True)
            return None

        sampled_question_df = available_questions.sample(1)
        row = sampled_question_df.iloc[0]
        csv_index = int(row["Index"])

        # mark assigned
        user_assignments[user_id].add(csv_index)

        questions_json_row = questions_json[csv_index]
        assert int(questions_json_row["Index"]) == csv_index, "CSV index and questions JSON index do not match"

        return {
            "question": str(questions_json_row["question"]),
            "default_response": str(questions_json_row["default_response"]),
            "truth": str(questions_json_row["truth"]),
            "csv_index": str(csv_index),
        }


def get_stats():
    """Get annotation statistics by reading the CSV."""
    try:
        with with_csv_lock():
            df = pd.read_csv(
                CONFIG["all_questions_metadata_csv_path"],
                usecols=[CONFIG["expert_dec_column"], "to_be_seen"],
                quoting=csv.QUOTE_ALL,
            )
            total = len(df)
            remaining = int(df["to_be_seen"].sum())
            matches = int((df[CONFIG["expert_dec_column"]] == CONFIG["match_value"]).sum())
            close_matches = int((df[CONFIG["expert_dec_column"]] == CONFIG["close_match_value"]).sum())
            vague_matches = int((df[CONFIG["expert_dec_column"]] == CONFIG["vague_match_value"]).sum())
            no_matches = int((df[CONFIG["expert_dec_column"]] == CONFIG["no_match_value"]).sum())
            annotated = total - remaining

            return {
                "total": total,
                "annotated": annotated,
                "remaining": remaining,
                "matches": matches,
                "close_matches": close_matches,
                "vague_matches": vague_matches,
                "no_matches": no_matches,
            }
    except Exception as e:
        print(f"❌ Error getting stats: {e}", flush=True)
        return None


def upload_csv_to_drive_best_effort():
    """Upload/update the metadata CSV in Google Drive (best-effort)."""
    if not (TOKEN_JSON and CLIENT_SECRET_JSON and FOLDER_ID):
        # silently skip if Drive isn't configured
        return

    service = get_drive_service()
    file_name = os.path.basename(CONFIG["all_questions_metadata_csv_path"])
    query = f"name='{file_name}' and '{FOLDER_ID}' in parents and trashed=false"

    existing_files = service.files().list(q=query, fields="files(id, name)").execute().get("files", [])
    media = MediaFileUpload(CONFIG["all_questions_metadata_csv_path"], resumable=True)

    if existing_files:
        file_id = existing_files[0]["id"]
        updated = service.files().update(fileId=file_id, media_body=media, fields="webViewLink").execute()
        print(f"🔄 Updated Drive file: {updated.get('webViewLink')}", flush=True)
    else:
        file_metadata = {"name": file_name, "parents": [FOLDER_ID]}
        created = service.files().create(body=file_metadata, media_body=media, fields="webViewLink").execute()
        print(f"✅ Uploaded new Drive file: {created.get('webViewLink')}", flush=True)


def save_annotation_to_csv(csv_index, annotation_value, comment="", user_id=None):
    """Save annotation/comment to CSV and upload updated CSV to Drive best-effort."""
    try:
        with with_csv_lock():
            df = pd.read_csv(
                CONFIG["all_questions_metadata_csv_path"],
                quoting=csv.QUOTE_ALL,
                dtype={"comments": "string"},
            )

            df.loc[df["Index"] == int(csv_index), CONFIG["expert_dec_column"]] = annotation_value
            df.loc[df["Index"] == int(csv_index), "to_be_seen"] = False
            df.loc[df["Index"] == int(csv_index), "comments"] = comment

            df.to_csv(CONFIG["all_questions_metadata_csv_path"], index=False, quoting=csv.QUOTE_ALL)

            # remove from all users
            idx = int(csv_index)
            for uid in list(user_assignments.keys()):
                user_assignments[uid].discard(idx)

        # Upload outside lock (avoid holding lock during network call)
        try:
            upload_csv_to_drive_best_effort()
        except Exception as e:
            print(f"⚠️ Drive upload skipped due to error: {e}", flush=True)

        return True
    except Exception as e:
        print(f"❌ Error saving annotation: {e}", flush=True)
        return False


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------
@app.route("/")
def serve_index():
    return send_from_directory(".", "index.html")


@app.route("/login")
def serve_login():
    return send_from_directory(".", "login.html")


@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(".", filename)


@app.route("/get_next_question", methods=["GET"])
def get_next_question():
    try:
        user_id = request.args.get("user", "1")
        question = get_slrt_random_unannotated_question(user_id)

        if question is None:
            stats = get_stats()
            if stats is None:
                return jsonify({"error": "Failed to get stats"}), 500
            return jsonify({
                "success": True,
                "completed": True,
                "message": "All questions have been annotated!",
                "stats": stats,
            })

        stats = get_stats()
        if stats is None:
            return jsonify({"error": "Failed to get stats"}), 500

        return jsonify({
            "success": True,
            "completed": False,
            "question": question,
            "stats": stats,
        })

    except Exception as e:
        print(f"❌ Error in get_next_question: {e}", flush=True)
        return jsonify({"error": str(e)}), 500


@app.route("/save_and_next", methods=["POST"])
def save_and_next():
    try:
        data = request.get_json()
        if not data or "annotation" not in data:
            return jsonify({"error": "No annotation provided"}), 400

        user_id = data.get("user_id", "1")
        annotation = data["annotation"]
        csv_index = int(float(data.get("csv_index")))
        comment = data.get("comment", "")

        if annotation == "match":
            annotation_value = CONFIG["match_value"]
        elif annotation == "close-match":
            annotation_value = CONFIG["close_match_value"]
        elif annotation == "vague-match":
            annotation_value = CONFIG["vague_match_value"]
        elif annotation == "no-match":
            annotation_value = CONFIG["no_match_value"]
        else:
            return jsonify({"error": "Invalid annotation value"}), 400

        if not save_annotation_to_csv(csv_index, annotation_value, comment, user_id):
            return jsonify({"error": "Failed to save annotation"}), 500

        question = get_slrt_random_unannotated_question(user_id)
        if question is None:
            stats = get_stats()
            if stats is None:
                return jsonify({"error": "Failed to get stats"}), 500
            return jsonify({
                "success": True,
                "completed": True,
                "message": "All questions have been annotated!",
                "stats": stats,
            })

        stats = get_stats()
        if stats is None:
            return jsonify({"error": "Failed to get stats"}), 500

        return jsonify({
            "success": True,
            "completed": False,
            "question": question,
            "stats": stats,
        })

    except Exception as e:
        print(f"❌ Error in save_and_next: {e}", flush=True)
        return jsonify({"error": str(e)}), 500


@app.route("/get_stats", methods=["GET"])
def get_stats_route():
    stats = get_stats()
    if stats is None:
        return jsonify({"error": "Failed to get stats"}), 500
    return jsonify({"success": True, "stats": stats})


@app.route("/health", methods=["GET"])
def health_check():
    is_valid, error_msg = validate_config()
    return jsonify({
        "status": "healthy" if is_valid else "misconfigured",
        "message": "Annotation server is running" if is_valid else error_msg,
        "config": {
            "all_questions_metadata_csv_path": CONFIG["all_questions_metadata_csv_path"],
            "all_questions_metadata_csv_exists": os.path.exists(CONFIG["all_questions_metadata_csv_path"]) if CONFIG["all_questions_metadata_csv_path"] else False,
            "all_questions_json_path": CONFIG["all_questions_json_path"],
            "all_questions_json_exists": os.path.exists(CONFIG["all_questions_json_path"]) if CONFIG["all_questions_json_path"] else False,
            "slrt_bounds_csv_path": CONFIG["slrt_bounds_csv_path"],
            "slrt_bounds_csv_exists": os.path.exists(CONFIG["slrt_bounds_csv_path"]) if CONFIG["slrt_bounds_csv_path"] else False,
        },
    })


# -----------------------------------------------------------------------------
# Local dev entry (Render uses gunicorn, not app.run())
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print_startup_banner()
    ok, _, _, _, err = validate_data_paths()
    if not ok:
        print(f"\n❌ Validation failed: {err}", flush=True)
        print("Fix DATA_CONFIG paths (local) or set ALL_QUESTIONS_* env vars (Render).", flush=True)
        raise SystemExit(1)

    port = int(os.getenv("PORT", "5001"))  # local default 5001, Render sets PORT
    print(f"🌐 Running dev server at http://localhost:{port}", flush=True)
    app.run(debug=True, host="0.0.0.0", port=port)
