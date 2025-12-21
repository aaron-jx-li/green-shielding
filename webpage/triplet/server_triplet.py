#!/usr/bin/env python3
"""
Simple Flask server for triplet annotation display.
"""

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import os
import pandas as pd
import csv

import json
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request

app = Flask(__name__)
CORS(app)

def get_drive_service():
    token_json = os.getenv("GOOGLE_TOKEN_JSON")
    scopes = os.getenv("SCOPES").split()

    if not token_json:
        raise Exception("Missing GOOGLE_TOKEN_JSON")

    creds = Credentials.from_authorized_user_info(json.loads(token_json), scopes)

    if creds.expired and creds.refresh_token:
        creds.refresh(Request())

    return build("drive", "v3", credentials=creds)

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

TRIPLET_SOURCE_CSV_PATH = TRIPLET_CSV_PATH

TRIPLET_RESULTS_CSV_PATH = os.path.abspath(
    os.path.join(
        BASE_DIR,
        "..",
        "annotation_manager",
        "ak_review_round0",
        "normalization",
        "rand_sample_triplet_results.csv"
    )
)

def init_results_csv():
    if os.path.exists(TRIPLET_RESULTS_CSV_PATH):
        return

    source_df = pd.read_csv(TRIPLET_SOURCE_CSV_PATH, quoting=csv.QUOTE_ALL)

    # Annotation columns only
    results_df = pd.DataFrame({
        "csv_index": source_df.index,
        "all_resp_match": 0,
        "none_resp_match": 0,
        "resp1_outlier": 0,
        "resp2_outlier": 0,
        "resp3_outlier": 0,
        "resp1_correct": 0,
        "resp2_correct": 0,
        "resp3_correct": 0,
        "user_annotated": -1
    })

    results_df.to_csv(TRIPLET_RESULTS_CSV_PATH, index=False, quoting=csv.QUOTE_ALL)
    print("✅ Initialized results CSV")


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
    source_df = pd.read_csv(TRIPLET_SOURCE_CSV_PATH, quoting=csv.QUOTE_ALL)
    results_df = pd.read_csv(TRIPLET_RESULTS_CSV_PATH, quoting=csv.QUOTE_ALL)

    unannotated_indices = results_df[results_df['user_annotated'] == -1]['csv_index']
    unannotated_df = source_df.loc[unannotated_indices]

    source_df = pd.read_csv(TRIPLET_SOURCE_CSV_PATH, quoting=csv.QUOTE_ALL)
    total = len(source_df)
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
    """Save annotation to CSV and get next question."""
    print("💾 /save_annotation endpoint called")
    try:
        data = request.get_json()
        
        if not data or 'annotation' not in data:
            return jsonify({'success': False, 'error': 'No annotation provided'}), 400
        
        csv_index = int(data.get('csv_index'))
        annotation = data.get('annotation')
        
        if csv_index is None:
            return jsonify({'success': False, 'error': 'No CSV index provided'}), 400
        
        print(f"📝 Saving annotation '{annotation}' for row {csv_index}")
        
        # Read the CSV
        df = pd.read_csv(TRIPLET_RESULTS_CSV_PATH, quoting=csv.QUOTE_ALL)
        matches = df[df['csv_index'] == csv_index]
        if len(matches) != 1:
            return jsonify({'success': False, 'error': 'Invalid csv_index'}), 400

        row = matches.index[0]

        
        if csv_index >= len(df):
            return jsonify({'success': False, 'error': f'Index {csv_index} out of range'}), 400
        
        # Initialize all annotation columns to 0
        df.loc[row, 'all_resp_match'] = 0
        df.loc[row, 'none_resp_match'] = 0
        df.loc[row, 'resp1_outlier'] = 0
        df.loc[row, 'resp2_outlier'] = 0
        df.loc[row, 'resp3_outlier'] = 0
        
        # Set the selected annotation to 1
        if annotation == 'all_resp_match':
            df.loc[row, 'all_resp_match'] = 1
        elif annotation == 'none_resp_match':
            df.loc[row, 'none_resp_match'] = 1
        elif annotation == 'resp1_outlier':
            df.loc[row, 'resp1_outlier'] = 1
        elif annotation == 'resp2_outlier':
            df.loc[row, 'resp2_outlier'] = 1
        elif annotation == 'resp3_outlier':
            df.loc[row, 'resp3_outlier'] = 1
        else:
            return jsonify({'success': False, 'error': 'Invalid annotation value'}), 400
        
        # Save correctness checkboxes
        df.loc[row, 'resp1_correct'] = int(data.get('resp1_correct', 0))
        df.loc[row, 'resp2_correct'] = int(data.get('resp2_correct', 0))
        df.loc[row, 'resp3_correct'] = int(data.get('resp3_correct', 0))

        print(f"✅ Correctness: resp1={df.loc[row, 'resp1_correct']}, resp2={df.loc[row, 'resp2_correct']}, resp3={df.loc[row, 'resp3_correct']}")
        # Set user_annotated to 1
        df.loc[row, 'user_annotated'] = 1
        
        # Save the CSV
        df.to_csv(TRIPLET_RESULTS_CSV_PATH, index=False, quoting=csv.QUOTE_ALL)
        print(f"✅ Annotation saved successfully for row {csv_index}")
        
        upload_results_to_drive()
        # Get next question
        result = get_next_triplet()
        if result.get('success'):
            return jsonify(result)
        else:
            return jsonify(result), 500
        
    except Exception as e:
        print(f"❌ Error in save_annotation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

def upload_results_to_drive():
    try:
        service = get_drive_service()
        file_name = os.path.basename(TRIPLET_RESULTS_CSV_PATH)
        folder_id = os.getenv("FOLDER_ID")

        query = f"name='{file_name}' and '{folder_id}' in parents and trashed=false"
        files = service.files().list(q=query, fields="files(id)").execute().get("files", [])

        media = MediaFileUpload(TRIPLET_RESULTS_CSV_PATH, resumable=True)

        if files:
            service.files().update(fileId=files[0]['id'], media_body=media).execute()
            print("🔄 Updated results CSV on Drive")
        else:
            service.files().create(
                body={"name": file_name, "parents": [folder_id]},
                media_body=media
            ).execute()
            print("✅ Uploaded results CSV to Drive")

    except Exception as e:
        print(f"⚠️ Drive sync failed: {e}")


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
    init_results_csv()
    print(f"📊 Source CSV: {TRIPLET_SOURCE_CSV_PATH}")
    print(f"📝 Results CSV: {TRIPLET_RESULTS_CSV_PATH}")
    app.run(debug=True, host='0.0.0.0', port=5001)

