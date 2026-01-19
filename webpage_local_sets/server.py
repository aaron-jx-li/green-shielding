#!/usr/bin/env python3
"""
Flask server for diagnosis annotation tool.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import os
import random
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configuration
DATA_PATH = "annotation_manager/data/sampled_data_HCM-3k.json"
ANNOTATIONS_PATH = "annotation_manager/data/annotations.json"
ASSIGNMENTS_PATH = "annotation_manager/data/user_assignments.json"

# Multi-user configuration
USERS = ['Dr. Kornblith', 'Dr. Bains']
USERS_PER_QUESTION = 2  # Number of users assigned to each question

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

def load_assignments():
    """Load question-to-users assignments."""
    if os.path.exists(ASSIGNMENTS_PATH):
        with open(ASSIGNMENTS_PATH, 'r') as f:
            return json.load(f)
    return {}

def save_assignments(assignments_dict):
    """Save question-to-users assignments."""
    with open(ASSIGNMENTS_PATH, 'w') as f:
        json.dump(assignments_dict, f, indent=2)

def initialize_assignments():
    """Initialize question assignments by randomly assigning each question to N users."""
    questions = load_data()
    assignments = load_assignments()
    
    # Check if assignments already exist for all questions
    question_indices = [str(q['__idx']) for q in questions]
    existing_indices = set(assignments.keys())
    new_indices = set(question_indices) - existing_indices
    
    if not new_indices:
        # All questions already assigned
        return assignments
    
    # For each unassigned question, randomly select N users
    for q_idx in new_indices:
        num_users = min(USERS_PER_QUESTION, len(USERS))
        assigned_users = random.sample(USERS, num_users)
        assignments[q_idx] = assigned_users
    
    save_assignments(assignments)
    return assignments

def get_user_next_question(user_id):
    """Get the next unannotated question for a specific user."""
    questions = load_data()
    annotations = load_annotations()
    assignments = load_assignments()
    
    # Ensure assignments are initialized
    if not assignments:
        assignments = initialize_assignments()
    
    # Find questions assigned to this user that haven't been annotated by this user
    for question in questions:
        q_idx = str(question['__idx'])
        
        # Check if this question is assigned to this user
        if q_idx in assignments and user_id in assignments[q_idx]:
            # Check if user has already annotated this question
            if q_idx not in annotations or user_id not in annotations[q_idx]:
                return question
    
    return None

def get_user_stats(user_id):
    """Get annotation statistics for a specific user."""
    questions = load_data()
    annotations = load_annotations()
    assignments = load_assignments()
    
    # Ensure assignments are initialized
    if not assignments:
        assignments = initialize_assignments()
    
    # Count questions assigned to this user
    total = 0
    annotated = 0
    
    for question in questions:
        q_idx = str(question['__idx'])
        if q_idx in assignments and user_id in assignments[q_idx]:
            total += 1
            if q_idx in annotations and user_id in annotations[q_idx]:
                annotated += 1
    
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

@app.route('/users', methods=['GET'])
def get_users():
    """Get list of available users."""
    return jsonify({
        'success': True,
        'users': USERS
    })

@app.route('/login', methods=['POST'])
def login():
    """Validate and set user ID."""
    try:
        data = request.get_json()
        user_id = data.get('user_id') if data else None
        
        if not user_id:
            return jsonify({'success': False, 'error': 'No user_id provided'}), 400
        
        if user_id not in USERS:
            return jsonify({'success': False, 'error': f'Invalid user_id. Must be one of: {USERS}'}), 400
        
        # Initialize assignments if needed
        initialize_assignments()
        
        return jsonify({
            'success': True,
            'user_id': user_id,
            'users': USERS
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/get_next_question', methods=['GET'])
def get_next_question():
    """Get the next unannotated question for a user."""
    try:
        user_id = request.args.get('user_id')
        
        if not user_id:
            return jsonify({'success': False, 'error': 'user_id parameter required'}), 400
        
        if user_id not in USERS:
            return jsonify({'success': False, 'error': f'Invalid user_id. Must be one of: {USERS}'}), 400
        
        question = get_user_next_question(user_id)
        
        if question is None:
            stats = get_user_stats(user_id)
            return jsonify({
                'success': True,
                'completed': True,
                'stats': stats
            })
        
        stats = get_user_stats(user_id)
        return jsonify({
            'success': True,
            'completed': False,
            'question': {
                'index': question['__idx'],
                'input': question['raw_input'],
                'plausible_set': question['ground_truth_space_majority']['plausible_set'],
                'highly_likely_set': question['ground_truth_space_majority']['highly_likely_set'],
                'cannot_miss_set': question['ground_truth_space_majority']['cannot_miss_set']
            },
            'stats': stats
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/save_annotation', methods=['POST'])
def save_annotation():
    """Save annotation and get next question."""
    try:
        data = request.get_json()
        
        if not data or '__idx' not in data:
            return jsonify({'success': False, 'error': 'No index provided'}), 400
        
        if 'user_id' not in data:
            return jsonify({'success': False, 'error': 'No user_id provided'}), 400
        
        user_id = data['user_id']
        if user_id not in USERS:
            return jsonify({'success': False, 'error': f'Invalid user_id. Must be one of: {USERS}'}), 400
        
        question_index = str(data['__idx'])
        annotation = {
            'not_plausible': data.get('not_plausible', []),
            'missing_plausible': data.get('missing_plausible', ''),
            'not_highly_likely': data.get('not_highly_likely', []),
            'missing_highly_likely': data.get('missing_highly_likely', ''),
            'not_cannot_miss': data.get('not_cannot_miss', []),
            'missing_cannot_miss': data.get('missing_cannot_miss', ''),
            'timestamp': datetime.now().isoformat()
        }
        
        annotations_dict = load_annotations()
        
        # Use nested structure: {question_index: {user_id: annotation}}
        if question_index not in annotations_dict:
            annotations_dict[question_index] = {}
        
        annotations_dict[question_index][user_id] = annotation
        save_annotations(annotations_dict)
        
        # Get next question for this user
        question = get_user_next_question(user_id)
        
        if question is None:
            stats = get_user_stats(user_id)
            return jsonify({
                'success': True,
                'completed': True,
                'stats': stats
            })
        
        stats = get_user_stats(user_id)
        return jsonify({
            'success': True,
            'completed': False,
            'question': {
                'index': question['__idx'],
                'input': question['raw_input'],
                'plausible_set': question['ground_truth_space_majority']['plausible_set'],
                'highly_likely_set': question['ground_truth_space_majority']['highly_likely_set'],
                'cannot_miss_set': question['ground_truth_space_majority']['cannot_miss_set']
            },
            'stats': stats
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    """Get user-specific statistics."""
    try:
        user_id = request.args.get('user_id')
        
        if not user_id:
            return jsonify({'success': False, 'error': 'user_id parameter required'}), 400
        
        if user_id not in USERS:
            return jsonify({'success': False, 'error': f'Invalid user_id. Must be one of: {USERS}'}), 400
        
        stats = get_user_stats(user_id)
        return jsonify({
            'success': True,
            'stats': stats
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/initialize_assignments', methods=['POST'])
def initialize_assignments_endpoint():
    """Initialize question assignments (idempotent)."""
    try:
        assignments = initialize_assignments()
        return jsonify({
            'success': True,
            'message': 'Assignments initialized',
            'total_questions': len(assignments)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

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

