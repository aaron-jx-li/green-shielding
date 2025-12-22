#!/usr/bin/env python3
"""
Flask server for dynamic annotation with server-side question selection.
This server manages question selection, tracks annotations, and updates the CSV file in real-time.
"""

from flask import Flask, request, jsonify, send_from_directory, redirect
from flask_cors import CORS
import json
import os
import random
from datetime import datetime
import pandas as pd
import csv

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.auth.transport.requests import Request

from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
CORS(app)  # Enable CORS for all routes

CLIENT_SECRET_JSON = os.getenv("GOOGLE_CLIENT_SECRET_JSON")
TOKEN_JSON = os.getenv("GOOGLE_TOKEN_JSON")
SCOPES = os.getenv("SCOPES", "https://www.googleapis.com/auth/drive.file").split()
REDIRECT_URI = os.getenv("REDIRECT_URI")
FOLDER_ID = os.getenv("FOLDER_ID")

# Global configuration - will be set by start_server.py or defaults
CONFIG = {
    'csv_path': None,
    'expert_dec_column': 'expert_dec',
    'unannotated_value': -1,
    'match_value': 3,
    'close_match_value': 2,
    'vague_match_value': 1,
    'no_match_value': 0
}

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

    print("✅ OAuth complete! Copy this token and add it to Render as GOOGLE_TOKEN_JSON:\n", flush=True)
    print(creds.to_json(), flush=True)

    return (
        "✅ Authentication successful! Check your Render logs and copy the token JSON into your Render environment as GOOGLE_TOKEN_JSON."
    )


def get_drive_service():
    """Build Drive service using env-stored token and refresh if needed."""
    if not TOKEN_JSON:
        raise Exception("❌ No token found. Run /authorize first.")

    creds = Credentials.from_authorized_user_info(json.loads(TOKEN_JSON), SCOPES)

    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())

        print("🔄 Access token refreshed successfully.", flush=True)

    return build('drive', 'v3', credentials=creds)

def validate_config():
    """Validate that the CSV file exists and has required columns."""
    try:
        if not CONFIG['all_questions_metadata_csv_path'] or not os.path.exists(CONFIG['all_questions_metadata_csv_path']):
            return False, "All questions metadata CSV file not found"
        
        if not CONFIG['all_questions_json_path'] or not os.path.exists(CONFIG['all_questions_json_path']):
            return False, "All questions JSON file not found"
        
        if not CONFIG['slrt_bounds_csv_path'] or not os.path.exists(CONFIG['slrt_bounds_csv_path']):
            return False, "SLRT bounds CSV file not found"
        
        return True, None
    except Exception as e:
        return False, str(e)

def get_slrt_random_unannotated_question():
    """Pick a random unannotated question and return only the necessary data."""
    questions_df = pd.read_csv(CONFIG['all_questions_metadata_csv_path'], quoting=csv.QUOTE_ALL)
    slrt_bounds_df = pd.read_csv(CONFIG['slrt_bounds_csv_path'], quoting=csv.QUOTE_ALL)
    questions_json = json.load(open(CONFIG['all_questions_json_path']))


    poss_cats = questions_df['category'].unique()
    for curr_cat in poss_cats:
        cat_questions_df = questions_df[questions_df['category'] == curr_cat]
        cat_num_seen = (~cat_questions_df['to_be_seen'].astype(bool)).sum()
        print("Category: ", curr_cat, "Number seen: ", cat_num_seen)
        min_samples = slrt_bounds_df.n.min()
        max_samples = slrt_bounds_df.n.max()
        if cat_num_seen < min_samples or cat_num_seen > max_samples:
            continue

        # print("csv_index:", csv_index, "len(questions_json):", len(questions_json))
        print("cat_num_seen:", cat_num_seen, "n values:", slrt_bounds_df['n'].tolist())

        row = slrt_bounds_df[slrt_bounds_df['n'] == cat_num_seen]
        if row.empty:
            continue

        cat_low_bound = row['lower'].iloc[0]
        cat_high_bound = row['upper'].iloc[0]

        num_incorrect = cat_questions_df[cat_questions_df['expert_dec'].isin([0, 1])].shape[0]
        print("Number incorrect: ", num_incorrect)
        print("Category low bound: ", cat_low_bound)
        print("Category high bound: ", cat_high_bound)
        if (num_incorrect >= cat_high_bound) or (num_incorrect <= cat_low_bound):
            print("Setting to_be_seen to False for category: ", curr_cat)
            questions_df.loc[questions_df['category'] == curr_cat, 'to_be_seen'] = False
    
    questions_df.to_csv(CONFIG['all_questions_metadata_csv_path'], index=False, quoting=csv.QUOTE_ALL)
    if sum(questions_df['to_be_seen']) == 0:
        print("No more need for annotation")
        return None
    sampled_question_df = questions_df[questions_df['to_be_seen'] == True].sample(1)


    row = sampled_question_df.iloc[0]
    csv_index = row['Index']

    print("csv_index:", csv_index, "len(questions_json):", len(questions_json))
    print("cat_num_seen:", cat_num_seen, "n values:", slrt_bounds_df['n'].tolist())

    questions_json_row = questions_json[csv_index]
    assert questions_json_row['Index'] == csv_index, "CSV index and questions JSON index do not match"
    question_data = {
        'question': str(questions_json_row['question']),
        'default_response': str(questions_json_row['default_response']),
        'truth': str(questions_json_row['truth']),
        'csv_index': str(csv_index)
    }
    return question_data

def get_stats():
    """Get annotation statistics by reading the CSV."""
    try:
        # Only read the expert_dec column for efficiency
        df = pd.read_csv(CONFIG['all_questions_metadata_csv_path'], usecols=[CONFIG['expert_dec_column'], 'to_be_seen'], quoting=csv.QUOTE_ALL)
        total = len(df)
        unannotated = sum(df['to_be_seen'])
        matches = len(df[df[CONFIG['expert_dec_column']] == CONFIG['match_value']])
        close_matches = len(df[df[CONFIG['expert_dec_column']] == CONFIG['close_match_value']])
        vague_matches = len(df[df[CONFIG['expert_dec_column']] == CONFIG['vague_match_value']])
        no_matches = len(df[df[CONFIG['expert_dec_column']] == CONFIG['no_match_value']])
        annotated = total - unannotated
        
        return {
            'total': total,
            'annotated': annotated,
            'remaining': unannotated,
            'matches': matches,
            'close_matches': close_matches,
            'vague_matches': vague_matches,
            'no_matches': no_matches
        }
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        return None

def save_annotation_to_csv(csv_index, annotation_value, comment=""):
    """Save annotation to the CSV file and upload/update it on Google Drive."""
    try:
        print("Saving annotation for index:", csv_index, "with value:", annotation_value)
        if comment:
            print("Saving comment: ", comment)
        df = pd.read_csv(
            CONFIG['all_questions_metadata_csv_path'],
            quoting=csv.QUOTE_ALL,
            dtype={"comments": "string"}
        )


        if sum(df['Index'] == int(csv_index)) != 1:
            print(f"❌ Error saving annotation: {df.shape[0]} rows found for index {csv_index}")
        else:
            print(f"✅ Found annotation row for CSV index {csv_index}: {annotation_value}")

        # Update annotation and mark as seen
        df.loc[df['Index'] == int(csv_index), CONFIG['expert_dec_column']] = annotation_value
        df.loc[df['Index'] == int(csv_index), "to_be_seen"] = False
        df.loc[df['Index'] == int(csv_index), "comments"] = comment

        # Save updated CSV locally
        df.to_csv(CONFIG['all_questions_metadata_csv_path'], index=False, quoting=csv.QUOTE_ALL)
        print(f"✅ Saved annotation locally for CSV row {csv_index}: {annotation_value}")
        if comment:
            print(f"✅ Saved comment for CSV row {csv_index}: {comment}")

        # Try uploading to Google Drive
        try:
            service = get_drive_service()
            file_name = os.path.basename(CONFIG['all_questions_metadata_csv_path'])
            query = f"name='{file_name}' and '{FOLDER_ID}' in parents and trashed=false"

            # Check if the file already exists in the Drive folder
            existing_files = service.files().list(q=query, fields="files(id, name)").execute().get('files', [])
            media = MediaFileUpload(CONFIG['all_questions_metadata_csv_path'], resumable=True)

            if existing_files:
                # Update existing file
                file_id = existing_files[0]['id']
                updated_file = service.files().update(
                    fileId=file_id,
                    media_body=media,
                    fields='id, name, webViewLink'
                ).execute()
                print(f"🔄 Updated existing Drive file: {updated_file['webViewLink']}")
            else:
                # Create a new file
                file_metadata = {'name': file_name, 'parents': [FOLDER_ID]}
                new_file = service.files().create(
                    body=file_metadata,
                    media_body=media,
                    fields='id, name, webViewLink'
                ).execute()
                print(f"✅ Uploaded new Drive file: {new_file['webViewLink']}")

        except Exception as e:
            print(f"⚠️ Google Drive upload skipped due to error: {e}")

        return True

    except Exception as e:
        print(f"❌ Error saving annotation: {e}")
        return False



@app.route('/')
def serve_index():
    """Serve the main HTML file."""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS, JSON)."""
    return send_from_directory('.', filename)

@app.route('/get_next_question', methods=['GET'])
def get_next_question():
    """Get the next unannotated question."""
    try:
        question = get_slrt_random_unannotated_question()

        if question is None:
            # No more unannotated questions
            stats = get_stats()
            if stats is None:
                return jsonify({'error': 'Failed to get stats'}), 500
                
            return jsonify({
                'success': True,
                'completed': True,
                'message': 'All questions have been annotated!',
                'stats': stats
            })
        
        # Get current stats
        stats = get_stats()
        if stats is None:
            return jsonify({'error': 'Failed to get stats'}), 500
        
        return jsonify({
            'success': True,
            'completed': False,
            'question': question,
            'stats': stats
        })
        
    except Exception as e:
        print(f"❌ Error in get_next_question: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/save_and_next', methods=['POST'])
def save_and_next():
    """Save the current annotation and get the next question."""
    try:
        data = request.get_json()
        
        if not data or 'annotation' not in data:
            return jsonify({'error': 'No annotation provided'}), 400
        
        annotation = data['annotation']
        csv_index = int(data['csv_index'])
        comment = data.get('comment', '')
        
        if csv_index is None:
            return jsonify({'error': 'No CSV index provided'}), 400
        
        # Convert annotation to value
        if annotation == 'match':
            annotation_value = CONFIG['match_value']
        elif annotation == 'close-match':
            annotation_value = CONFIG['close_match_value']
        elif annotation == 'vague-match':
            annotation_value = CONFIG['vague_match_value']
        elif annotation == 'no-match':
            annotation_value = CONFIG['no_match_value']
        else:
            return jsonify({'error': 'Invalid annotation value'}), 400
        
        # Save annotation
        if not save_annotation_to_csv(csv_index, annotation_value, comment):
            return jsonify({'error': 'Failed to save annotation'}), 500
        
        # Get next question
        question = get_slrt_random_unannotated_question()
        
        if question is None:
            # No more unannotated questions
            print("No more unannotated questions")
            stats = get_stats()
            if stats is None:
                return jsonify({'error': 'Failed to get stats'}), 500
                
            return jsonify({
                'success': True,
                'completed': True,
                'message': 'All questions have been annotated!',
                'stats': stats
            })
        
        # Get current stats
        print("Getting stats after saving annotation")
        stats = get_stats()
        if stats is None:
            return jsonify({'error': 'Failed to get stats'}), 500
        
        print("Stats after saving annotation: ", stats)
        return jsonify({
            'success': True,
            'completed': False,
            'question': question,
            'stats': stats
        })
        
    except Exception as e:
        print(f"❌ Error in save_and_next: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/get_stats', methods=['GET'])
def get_stats_route():
    """Get current annotation statistics."""
    try:
        stats = get_stats()
        
        if stats is None:
            return jsonify({'error': 'Failed to get stats'}), 500
        
        return jsonify({
            'success': True,
            'stats': stats
        })
        
    except Exception as e:
        print(f"❌ Error in get_stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    is_valid, error_msg = validate_config()
    
    return jsonify({
        'status': 'healthy' if is_valid else 'misconfigured',
        'message': 'Annotation server is running' if is_valid else error_msg,
        'config': {
            'all_questions_metadata_csv_path': CONFIG['all_questions_metadata_csv_path'],
            'all_questions_metadata_csv_exists': os.path.exists(CONFIG['all_questions_metadata_csv_path']) if CONFIG['all_questions_metadata_csv_path'] else False,
            'all_questions_json_path': CONFIG['all_questions_json_path'],
            'all_questions_json_exists': os.path.exists(CONFIG['all_questions_json_path']) if CONFIG['all_questions_json_path'] else False,
            'slrt_bounds_csv_path': CONFIG['slrt_bounds_csv_path'],
            'slrt_bounds_csv_exists': os.path.exists(CONFIG['slrt_bounds_csv_path']) if CONFIG['slrt_bounds_csv_path'] else False
        }
    })

def configure(all_questions_metadata_csv_path, all_questions_json_path, slrt_bounds_csv_path, expert_dec_column='expert_dec', 
              unannotated_value=-1, match_value=3, close_match_value=2, vague_match_value=1, no_match_value=0):
    """Configure the server with data paths and column settings."""
    CONFIG['all_questions_metadata_csv_path'] = all_questions_metadata_csv_path
    CONFIG['all_questions_json_path'] = all_questions_json_path
    CONFIG['slrt_bounds_csv_path'] = slrt_bounds_csv_path
    CONFIG['expert_dec_column'] = expert_dec_column
    CONFIG['unannotated_value'] = unannotated_value
    CONFIG['match_value'] = match_value
    CONFIG['close_match_value'] = close_match_value
    CONFIG['vague_match_value'] = vague_match_value
    CONFIG['no_match_value'] = no_match_value
    
    print(f"🔧 Server configured:")
    print(f"   All questions metadata CSV: {all_questions_metadata_csv_path}")
    print(f"   All questions JSON: {all_questions_json_path}")
    print(f"   Expert column: {expert_dec_column}")
    
    # Validate configuration
    is_valid, error_msg = validate_config()
    if not is_valid:
        print(f"❌ Configuration error: {error_msg}")
        return False
    
    # Print initial stats
    stats = get_stats()
    if stats:
        print(f"✅ Loaded CSV with {stats['total']} total questions")
        print(f"   Unannotated: {stats['remaining']}")
        print(f"   Annotated: {stats['annotated']}")
        print(f"      - Accurate Matches: {stats['matches']}")
        print(f"      - Close Matches: {stats['close_matches']}")
        print(f"      - Vague Matches: {stats['vague_matches']}")
        print(f"      - No Matches: {stats['no_matches']}")
    
    return True

if __name__ == '__main__':
    print("🚀 Starting Dynamic Annotation Server...")
    print("⚠️  Note: Use start_server.py to properly configure data paths")
    print("🌐 Server will be available at: http://localhost:5001")
    print("📊 Health check: http://localhost:5001/health")
    print("\n" + "="*50)
    
    app.run(debug=True, host='0.0.0.0', port=5001)
