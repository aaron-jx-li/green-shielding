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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.environ.get("DATA_PATH",os.path.join(BASE_DIR, "data", "sampled_data_HCM-3k.json"),)
ANNOTATIONS_PATH = os.environ.get("ANNOTATIONS_PATH",os.path.join(BASE_DIR, "data", "annotations.json"),)

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
    return send_from_directory(BASE_DIR, 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory(BASE_DIR, filename)

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

