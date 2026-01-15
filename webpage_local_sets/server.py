#!/usr/bin/env python3
"""
Flask server for diagnosis annotation tool.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configuration
DATA_PATH = "/Users/sancheznicolas/Documents/Research/GreenTeam/green_shielding3/green-shielding/webpage_local_sets/annotation_manager/data/sampled_data_HCM-3k.json"
ANNOTATIONS_PATH = "/Users/sancheznicolas/Documents/Research/GreenTeam/green_shielding3/green-shielding/webpage_local_sets/annotation_manager/data/annotations.json"

# Global data
questions_data = None
annotations = {}

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
    """Save annotations to JSON file."""
    with open(ANNOTATIONS_PATH, 'w') as f:
        json.dump(annotations_dict, f, indent=2)

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
    """Serve the main HTML file."""
    return send_from_directory('.', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files (CSS, JS)."""
    return send_from_directory('.', filename)

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
        dx_set = question['metrics']['extracted_diagnoses']
        stats = get_stats()
        return jsonify({
            'success': True,
            'completed': False,
            'question': {
                'index': question['index'],
                'input': question['input'],
                'plausible_set': question['judge_dx_space']['plausible_set'],
                'highly_likely_set': question['judge_dx_space']['highly_likely_set'],
                'dx_set': dx_set
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
        
        # Process dx_plausible_matches into a list of all pairs with match status
        dx_plausible_matches_raw = data.get('dx_plausible_matches', {})
        dx_plausible_pairs = []
        
        # Get all unique dx_set and plausible_set from the current question
        # We need to load the question to get the full sets
        questions = load_data()
        current_question = next((q for q in questions if str(q['index']) == question_index), None)
        
        if current_question:
            dx_set = current_question['metrics']['extracted_diagnoses']
            plausible_set = current_question['judge_dx_space']['plausible_set']
            
            # Create pairs for all combinations
            for dx in dx_set:
                for plausible in plausible_set:
                    # Check if this pair was matched (default to False if not in matches)
                    is_matched = dx_plausible_matches_raw.get(dx, {}).get(plausible, False)
                    dx_plausible_pairs.append({
                        'dx': dx,
                        'plausible': plausible,
                        'matched': is_matched
                    })
        
        annotation = {
            'index': question_index,
            'not_plausible': data.get('not_plausible', []),
            'missing_plausible': data.get('missing_plausible', ''),
            'not_highly_likely': data.get('not_highly_likely', []),
            'missing_highly_likely': data.get('missing_highly_likely', ''),
            'dx_plausible_pairs': dx_plausible_pairs,
            'timestamp': datetime.now().isoformat()
        }
        
        annotations_dict = load_annotations()
        annotations_dict[question_index] = annotation
        save_annotations(annotations_dict)
        
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
        dx_set = question['metrics']['extracted_diagnoses']
        return jsonify({
            'success': True,
            'completed': False,
            'question': {
                'index': question['index'],
                'input': question['input'],
                'plausible_set': question['judge_dx_space']['plausible_set'],
                'highly_likely_set': question['judge_dx_space']['highly_likely_set'],
                'dx_set': dx_set
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
    
    app.run(debug=True, host='0.0.0.0', port=5002)

