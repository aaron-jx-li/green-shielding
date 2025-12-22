#!/usr/bin/env python3
"""
Simple Flask server for triplet annotation display.
"""

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import os
import pandas as pd
import csv
from datetime import datetime

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request
import json


app = Flask(__name__)
CORS(app)

# Path to the triplet CSV file ### MUST CHANGE ####
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEBPAGE_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))

TRIPLET_CSV_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "annotation_manager",
        "ak_review_round0",
        "normalization",
        "rand_sample_triplet.csv"
    )
)

RESULTS_CSV_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "annotation_manager",
        "ak_review_round0",
        "normalization",
        "triplet_annotations_results.csv"
    )
)

def ensure_results_csv():
    if not os.path.exists(RESULTS_CSV_PATH):
        df = pd.DataFrame(columns=[
            "timestamp",
            "csv_index",
            "question_og",
            "annotation",
            "resp1_correct",
            "resp2_correct",
            "resp3_correct"
        ])
        df.to_csv(RESULTS_CSV_PATH, index=False, quoting=csv.QUOTE_ALL)

def upload_results_to_drive():
    if not os.getenv("GOOGLE_TOKEN_JSON"):
        print("⚠️ Drive upload skipped (no token)")
        return

    creds = Credentials.from_authorized_user_info(
        json.loads(os.getenv("GOOGLE_TOKEN_JSON")),
        ["https://www.googleapis.com/auth/drive.file"]
    )

    if creds.expired and creds.refresh_token:
        creds.refresh(Request())

    service = build("drive", "v3", credentials=creds)

    file_name = os.path.basename(RESULTS_CSV_PATH)
    media = MediaFileUpload(RESULTS_CSV_PATH, resumable=True)

    query = f"name='{file_name}' and '{os.getenv('FOLDER_ID')}' in parents and trashed=false"
    existing = service.files().list(q=query, fields="files(id)").execute().get("files", [])

    if existing:
        service.files().update(
            fileId=existing[0]["id"],
            media_body=media
        ).execute()
        print("🔄 Updated Drive results CSV")
    else:
        service.files().create(
            body={"name": file_name, "parents": [os.getenv("FOLDER_ID")]},
            media_body=media
        ).execute()
        print("✅ Uploaded new Drive results CSV")

def get_next_triplet():
    """Get a random triplet from the CSV. Returns dict or None if completed."""
    print(f"📂 Checking CSV path: {TRIPLET_CSV_PATH}")
    if not os.path.exists(TRIPLET_CSV_PATH):
        print(f"❌ CSV file not found at {TRIPLET_CSV_PATH}")
        return {'success': False, 'error': f'CSV file not found at {TRIPLET_CSV_PATH}'}
    
    print("📖 Reading CSV file...")
    df = pd.read_csv(TRIPLET_CSV_PATH, quoting=csv.QUOTE_ALL)
    print(f"✅ CSV loaded with {len(df)} rows")
    
    if len(df) == 0:
        print("❌ CSV file is empty")
        return {'success': False, 'error': 'CSV file is empty'}
    
    # Get unannotated rows
    if os.path.exists(RESULTS_CSV_PATH):
        results_df = pd.read_csv(RESULTS_CSV_PATH)
        annotated_indices = set(results_df["csv_index"].astype(int))
    else:
        annotated_indices = set()

    unannotated_df = df[~df.index.isin(annotated_indices)]

    total = len(df)
    unannotated = len(unannotated_df)
    annotated = total - unannotated
    
    if len(unannotated_df) == 0:
        print("✅ All questions have been annotated!")
        return {
            'success': True,
            'completed': True,
            'message': 'All questions have been annotated!',
            'stats': {
                'total': total,
                'annotated': annotated,
                'unannotated': unannotated
            }
        }
    
    # Get a random row
    print("🎲 Sampling random row...")
    sampled_row = unannotated_df.sample(1).iloc[0]
    csv_index = sampled_row.name  # Get the row index
    print(f"✅ Row sampled successfully (index: {csv_index}) from unannotated {len(unannotated_df)} questions")
    
    # Return the question and the three randomized responses with their labels
    result = {
        'success': True,
        'completed': False,
        'csv_index': int(csv_index),
        'question_og': str(sampled_row.get('question_og', 'N/A')),
        'response1_label': str(sampled_row.get('Response1_label', 'N/A')),
        'response1_value': str(sampled_row.get('Response1_value', 'N/A')),
        'response2_label': str(sampled_row.get('Response2_label', 'N/A')),
        'response2_value': str(sampled_row.get('Response2_value', 'N/A')),
        'response3_label': str(sampled_row.get('Response3_label', 'N/A')),
        'response3_value': str(sampled_row.get('Response3_value', 'N/A')),
        'stats': {
            'total': total,
            'annotated': annotated,
            'unannotated': unannotated
        }
    }
    print("✅ Returning data")
    return result

@app.route('/get_triplet', methods=['GET'])
def get_triplet():
    """Get a random triplet from the CSV."""
    print("🔍 /get_triplet endpoint called")
    try:
        result = get_next_triplet()
        if result.get('success') and not result.get('completed'):
            return jsonify(result)
        elif result.get('completed'):
            return jsonify(result)
        else:
            return jsonify(result), 404
    except Exception as e:
        print(f"❌ Error in get_triplet: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/')
def serve_index():
    """Serve the triplet HTML file."""
    return send_from_directory('.', 'index_triplet.html')


@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    print("💾 /save_annotation endpoint called")
    try:
        data = request.get_json()

        csv_index = int(data['csv_index'])
        annotation = data['annotation']

        ensure_results_csv()

        row = {
            "timestamp": datetime.utcnow().isoformat(),
            "csv_index": csv_index,
            "question_og": data.get("question_og", ""),
            "annotation": annotation,
            "resp1_correct": int(data.get("resp1_correct", 0)),
            "resp2_correct": int(data.get("resp2_correct", 0)),
            "resp3_correct": int(data.get("resp3_correct", 0)),
        }

        df = pd.DataFrame([row])
        df.to_csv(
            RESULTS_CSV_PATH,
            mode="a",
            header=False,
            index=False,
            quoting=csv.QUOTE_ALL
        )

        print("✅ Annotation appended")


        upload_results_to_drive()
        return jsonify(get_next_triplet())

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({'success': False, 'error': str(e)}), 500

# Serve CSS from webpage/
@app.route('/static/css/<path:filename>')
def serve_css(filename):
    return send_from_directory(WEBPAGE_DIR, filename)

# Serve JS from webpage/triplet/
@app.route('/static/js/<path:filename>')
def serve_js(filename):
    return send_from_directory(os.path.join(WEBPAGE_DIR, "triplet"), filename)

if __name__ == '__main__':
    print("🚀 Starting Triplet Annotation Server...")
    print(f"📊 CSV path: {TRIPLET_CSV_PATH}")
    ensure_results_csv()
    print("🌐 Server will be available at: http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)