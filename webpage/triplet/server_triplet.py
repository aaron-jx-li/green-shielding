#!/usr/bin/env python3
"""
Simple Flask server for triplet annotation display.
"""

from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import os
import pandas as pd
import csv

app = Flask(__name__)
CORS(app)

# Path to the triplet CSV file ### MUST CHANGE ####
TRIPLET_CSV_PATH = "/Users/sancheznicolas/Documents/Research/GreenTeam/green_shield_eval/green-shielding/webpage_local/annotation_manager/ak_review_round0/normalization/rand_sample_triplet.csv"

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
    unannotated_df = df[df['user_annotated'] == -1]
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
        df = pd.read_csv(TRIPLET_CSV_PATH, quoting=csv.QUOTE_ALL)
        
        if csv_index >= len(df):
            return jsonify({'success': False, 'error': f'Index {csv_index} out of range'}), 400
        
        # Initialize all annotation columns to 0
        df.loc[csv_index, 'all_resp_match'] = 0
        df.loc[csv_index, 'none_resp_match'] = 0
        df.loc[csv_index, 'resp1_outlier'] = 0
        df.loc[csv_index, 'resp2_outlier'] = 0
        df.loc[csv_index, 'resp3_outlier'] = 0
        
        # Set the selected annotation to 1
        if annotation == 'all_resp_match':
            df.loc[csv_index, 'all_resp_match'] = 1
        elif annotation == 'none_resp_match':
            df.loc[csv_index, 'none_resp_match'] = 1
        elif annotation == 'resp1_outlier':
            df.loc[csv_index, 'resp1_outlier'] = 1
        elif annotation == 'resp2_outlier':
            df.loc[csv_index, 'resp2_outlier'] = 1
        elif annotation == 'resp3_outlier':
            df.loc[csv_index, 'resp3_outlier'] = 1
        else:
            return jsonify({'success': False, 'error': 'Invalid annotation value'}), 400
        
        # Save correctness checkboxes
        df.loc[csv_index, 'resp1_correct'] = int(data.get('resp1_correct', 0))
        df.loc[csv_index, 'resp2_correct'] = int(data.get('resp2_correct', 0))
        df.loc[csv_index, 'resp3_correct'] = int(data.get('resp3_correct', 0))
        
        print(f"✅ Correctness: resp1={df.loc[csv_index, 'resp1_correct']}, resp2={df.loc[csv_index, 'resp2_correct']}, resp3={df.loc[csv_index, 'resp3_correct']}")
        
        # Set user_annotated to 1
        df.loc[csv_index, 'user_annotated'] = 1
        
        # Save the CSV
        df.to_csv(TRIPLET_CSV_PATH, index=False, quoting=csv.QUOTE_ALL)
        print(f"✅ Annotation saved successfully for row {csv_index}")
        
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

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS)."""
    return send_from_directory('.', filename)

if __name__ == '__main__':
    print("🚀 Starting Triplet Annotation Server...")
    print(f"📊 CSV path: {TRIPLET_CSV_PATH}")
    print("🌐 Server will be available at: http://localhost:5001")
    app.run(debug=True, host='0.0.0.0', port=5001)
