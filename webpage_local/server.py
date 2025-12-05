#!/usr/bin/env python3
"""
Flask server for dynamic annotation with server-side question selection.
This server manages question selection, tracks annotations, and updates the CSV file in real-time.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
import random
from datetime import datetime
import pandas as pd
import csv

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global configuration - will be set by start_server.py or defaults
CONFIG = {
    'csv_path': None,
    'expert_dec_column': 'expert_dec',
    'unannotated_value': -1,
    'match_value': 3,
    'close_match_value': 2,
    'vague_match_value': 1,
    'no_match_value': 0,
    'normalization_csv_path': None,
    'user_annotation_column': 'user_annotation'
}

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

def validate_norm_config():
    """Validate that the normalization CSV file exists."""
    try:
        if not CONFIG['normalization_csv_path'] or not os.path.exists(CONFIG['normalization_csv_path']):
            return False, "Normalization CSV file not found"
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
        cat_num_seen = sum(~cat_questions_df['to_be_seen'])
        print("Category: ", curr_cat, "Number seen: ", cat_num_seen)
        min_samples = slrt_bounds_df.n.min()
        max_samples = slrt_bounds_df.n.max()
        if cat_num_seen < min_samples or cat_num_seen > max_samples:
            continue
        cat_low_bound = slrt_bounds_df[slrt_bounds_df['n'] == cat_num_seen]['lower'].values[0]
        cat_high_bound = slrt_bounds_df[slrt_bounds_df['n'] == cat_num_seen]['upper'].values[0]
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
    """Save annotation and comment to the CSV file."""
    try:
        print("Saving annotation for index: ", csv_index, " with value: ", annotation_value)
        if comment:
            print("Saving comment: ", comment)
        # Read the entire CSV
        df = pd.read_csv(CONFIG['all_questions_metadata_csv_path'], quoting=csv.QUOTE_ALL)
        
        if sum(df['Index'] == int(csv_index)) != 1:
            print(f"❌ Error saving annotation: {df.shape[0]} rows found for index {csv_index}")
        else:
            print(f"✅ properly found annotation for CSV row {csv_index}: {annotation_value}")
        
        row_to_update = df[df['Index'] == int(csv_index)].iloc[0]
        # Update the specific row with the annotation value (0, 1, 2, or 3)
        df.loc[df['Index'] == int(csv_index), CONFIG['expert_dec_column']] = annotation_value
        df.loc[df['Index'] == int(csv_index), "to_be_seen"] = False
        # Update the comment column
        df.loc[df['Index'] == int(csv_index), "comments"] = comment
        
        # Save back to file
        df.to_csv(CONFIG['all_questions_metadata_csv_path'], index=False, quoting=csv.QUOTE_ALL)
        
        print(f"✅ Saved annotation for CSV row {csv_index}: {annotation_value}")
        if comment:
            print(f"✅ Saved comment for CSV row {csv_index}")
        return True
    except Exception as e:
        print(f"❌ Error saving annotation: {e}")
        return False

# Normalization functions
def get_random_unannotated_question_norm():
    """Pick a random unannotated normalization question."""
    try:
        df = pd.read_csv(CONFIG['normalization_csv_path'], quoting=csv.QUOTE_ALL)
        
        # Filter for unannotated rows (user_annotation is -1 or NaN)
        unannotated_df = df[(df[CONFIG['user_annotation_column']].isna()) | (df[CONFIG['user_annotation_column']] == -1)]
        
        if len(unannotated_df) == 0:
            return None
        
        # Sample a random row
        sampled_row = unannotated_df.sample(1).iloc[0]
        
        # Get the index (row number in the CSV, 0-based)
        csv_index = sampled_row.name
        
        question_data = {
            'question_og': str(sampled_row.get('question_og', 'N/A')),
            'model_response_og': str(sampled_row.get('model_response_og', 'N/A')),
            'model_response_conv': str(sampled_row.get('model_response_conv', 'N/A')),
            'reference_diagnosis_conv': str(sampled_row.get('reference_diagnosis_conv', 'N/A')),
            'csv_index': str(csv_index)
        }
        return question_data
    except Exception as e:
        print(f"❌ Error getting normalization question: {e}")
        return None

def get_stats_norm():
    """Get normalization annotation statistics."""
    try:
        df = pd.read_csv(CONFIG['normalization_csv_path'], quoting=csv.QUOTE_ALL)
        total = len(df)
        
        # Count unannotated (user_annotation is -1 or NaN)
        unannotated = len(df[(df[CONFIG['user_annotation_column']].isna()) | (df[CONFIG['user_annotation_column']] == -1)])
        annotated = total - unannotated
        
        # Count original (0) and normalized (1) selections
        original_selected = len(df[df[CONFIG['user_annotation_column']] == 0])
        normalized_selected = len(df[df[CONFIG['user_annotation_column']] == 1])
        
        return {
            'total': total,
            'annotated': annotated,
            'remaining': unannotated,
            'original_selected': original_selected,
            'normalized_selected': normalized_selected
        }
    except Exception as e:
        print(f"❌ Error getting normalization stats: {e}")
        return None

def save_annotation_to_csv_norm(csv_index, annotation_value, comment=""):
    """Save normalization annotation to the CSV file."""
    try:
        print("Saving normalization annotation for index: ", csv_index, " with value: ", annotation_value)
        if comment:
            print("Saving comment: ", comment)
        
        # Read the entire CSV
        df = pd.read_csv(CONFIG['normalization_csv_path'], quoting=csv.QUOTE_ALL)
        
        # csv_index is the row number (0-based)
        if int(csv_index) >= len(df):
            print(f"❌ Error saving annotation: index {csv_index} out of range")
            return False
        
        # Update the user_annotation column
        df.loc[int(csv_index), CONFIG['user_annotation_column']] = annotation_value
        
        # Update comment if comment column exists
        if 'comments' in df.columns:
            df.loc[int(csv_index), 'comments'] = comment
        
        # Save back to file
        df.to_csv(CONFIG['normalization_csv_path'], index=False, quoting=csv.QUOTE_ALL)
        
        print(f"✅ Saved normalization annotation for CSV row {csv_index}: {annotation_value}")
        if comment:
            print(f"✅ Saved comment for CSV row {csv_index}")
        return True
    except Exception as e:
        print(f"❌ Error saving normalization annotation: {e}")
        return False

@app.route('/')
def serve_index():
    """Serve the main HTML file."""
    return send_from_directory('.', 'index.html')

@app.route('/norm')
def serve_index_norm():
    """Serve the normalization HTML file."""
    return send_from_directory('.', 'index_norm.html')

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
        csv_index = int(float(data.get('csv_index')))
        comment = data.get('comment', '')  # Get comment, default to empty string if not provided
        
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
        
        # Save annotation and comment
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

@app.route('/get_next_question_norm', methods=['GET'])
def get_next_question_norm():
    """Get the next unannotated normalization question."""
    try:
        question = get_random_unannotated_question_norm()

        if question is None:
            # No more unannotated questions
            stats = get_stats_norm()
            if stats is None:
                return jsonify({'error': 'Failed to get stats'}), 500
                
            return jsonify({
                'success': True,
                'completed': True,
                'message': 'All questions have been annotated!',
                'stats': stats
            })
        
        # Get current stats
        stats = get_stats_norm()
        if stats is None:
            return jsonify({'error': 'Failed to get stats'}), 500
        
        return jsonify({
            'success': True,
            'completed': False,
            'question': question,
            'stats': stats
        })
        
    except Exception as e:
        print(f"❌ Error in get_next_question_norm: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/save_and_next_norm', methods=['POST'])
def save_and_next_norm():
    """Save the current normalization annotation and get the next question."""
    try:
        data = request.get_json()
        
        if not data or 'annotation' not in data:
            return jsonify({'error': 'No annotation provided'}), 400
        
        annotation = data['annotation']
        csv_index = int(float(data.get('csv_index')))
        comment = data.get('comment', '')
        
        if csv_index is None:
            return jsonify({'error': 'No CSV index provided'}), 400
        
        # Convert annotation to value: original = 0, normalized = 1
        if annotation == 'original':
            annotation_value = 0
        elif annotation == 'normalized':
            annotation_value = 1
        else:
            return jsonify({'error': 'Invalid annotation value'}), 400
        
        # Save annotation and comment
        if not save_annotation_to_csv_norm(csv_index, annotation_value, comment):
            return jsonify({'error': 'Failed to save annotation'}), 500
        
        # Get next question
        question = get_random_unannotated_question_norm()
        
        if question is None:
            # No more unannotated questions
            print("No more unannotated normalization questions")
            stats = get_stats_norm()
            if stats is None:
                return jsonify({'error': 'Failed to get stats'}), 500
                
            return jsonify({
                'success': True,
                'completed': True,
                'message': 'All questions have been annotated!',
                'stats': stats
            })
        
        # Get current stats
        print("Getting stats after saving normalization annotation")
        stats = get_stats_norm()
        if stats is None:
            return jsonify({'error': 'Failed to get stats'}), 500
        
        print("Stats after saving normalization annotation: ", stats)
        return jsonify({
            'success': True,
            'completed': False,
            'question': question,
            'stats': stats
        })
        
    except Exception as e:
        print(f"❌ Error in save_and_next_norm: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    is_valid, error_msg = validate_config()
    is_valid_norm, error_msg_norm = validate_norm_config()
    
    return jsonify({
        'status': 'healthy' if (is_valid and is_valid_norm) else 'misconfigured',
        'message': 'Annotation server is running' if (is_valid and is_valid_norm) else (error_msg or error_msg_norm),
        'config': {
            'all_questions_metadata_csv_path': CONFIG['all_questions_metadata_csv_path'],
            'all_questions_metadata_csv_exists': os.path.exists(CONFIG['all_questions_metadata_csv_path']) if CONFIG['all_questions_metadata_csv_path'] else False,
            'all_questions_json_path': CONFIG['all_questions_json_path'],
            'all_questions_json_exists': os.path.exists(CONFIG['all_questions_json_path']) if CONFIG['all_questions_json_path'] else False,
            'slrt_bounds_csv_path': CONFIG['slrt_bounds_csv_path'],
            'slrt_bounds_csv_exists': os.path.exists(CONFIG['slrt_bounds_csv_path']) if CONFIG['slrt_bounds_csv_path'] else False,
            'normalization_csv_path': CONFIG['normalization_csv_path'],
            'normalization_csv_exists': os.path.exists(CONFIG['normalization_csv_path']) if CONFIG['normalization_csv_path'] else False
        }
    })

def configure(all_questions_metadata_csv_path, all_questions_json_path, slrt_bounds_csv_path, expert_dec_column='expert_dec', 
              unannotated_value=-1, match_value=3, close_match_value=2, vague_match_value=1, no_match_value=0,
              normalization_csv_path=None):
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
    CONFIG['normalization_csv_path'] = normalization_csv_path
    
    print(f"🔧 Server configured:")
    print(f"   All questions metadata CSV: {all_questions_metadata_csv_path}")
    print(f"   All questions JSON: {all_questions_json_path}")
    if normalization_csv_path:
        print(f"   Normalization CSV: {normalization_csv_path}")
    print(f"   Expert column: {expert_dec_column}")
    
    # Validate configuration
    is_valid, error_msg = validate_config()
    if not is_valid:
        print(f"❌ Configuration error: {error_msg}")
        return False
    
    # Validate normalization config if provided
    if normalization_csv_path:
        is_valid_norm, error_msg_norm = validate_norm_config()
        if not is_valid_norm:
            print(f"❌ Normalization configuration error: {error_msg_norm}")
            return False
        
        # Print normalization stats
        stats_norm = get_stats_norm()
        if stats_norm:
            print(f"✅ Loaded normalization CSV with {stats_norm['total']} total questions")
            print(f"   Unannotated: {stats_norm['remaining']}")
            print(f"   Annotated: {stats_norm['annotated']}")
            print(f"      - Original Selected: {stats_norm['original_selected']}")
            print(f"      - Normalized Selected: {stats_norm['normalized_selected']}")
    
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
