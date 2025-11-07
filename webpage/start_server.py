#!/usr/bin/env python3
"""
Startup script for the dynamic annotation server.
This will install dependencies, configure data paths, and start the Flask server.
"""

import subprocess
import sys
import os

# ========== CONFIGURATION ==========
# Set your CSV data file path here
DATA_CONFIG = {
    'all_questions_metadata_csv_path': 'annotation_manager/ak_review_round0/all_questions_metadata.csv',
    'all_questions_json_path': 'annotation_manager/ak_review_round0/all_questions.json',
    'slrt_bounds_csv_path': 'annotation_manager/ak_review_round0/slrt_test_low_upper_bound.csv',
    'expert_dec_column': 'expert_dec',
    'unannotated_value': -1,
    'match_value': 3,
    'close_match_value': 2,
    'vague_match_value': 1,
    'no_match_value': 0
}
# ===================================

def install_requirements():
    """Install required Python packages."""
    print("📦 Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Dependencies installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False

def validate_data_paths():
    """Validate that the configured CSV file exists."""
    all_questions_metadata_csv_path = DATA_CONFIG['all_questions_metadata_csv_path']
    all_questions_json_path = DATA_CONFIG['all_questions_json_path']
    slrt_bounds_csv_path = DATA_CONFIG['slrt_bounds_csv_path']

    # Convert relative paths to absolute
    if not os.path.isabs(all_questions_metadata_csv_path):
        all_questions_metadata_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), all_questions_metadata_csv_path))
    

    if not os.path.exists(all_questions_json_path):
        print(f"❌ All questions JSON file not found: {all_questions_json_path}")
        return False, None

    if not os.path.exists(all_questions_metadata_csv_path):
        print(f"❌ All questions metadata CSV file not found: {all_questions_metadata_csv_path}")
        return False, None
    
    if not os.path.isabs(slrt_bounds_csv_path):
        slrt_bounds_csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), slrt_bounds_csv_path))
    if not os.path.exists(slrt_bounds_csv_path):
        print(f"❌ SLRT bounds CSV file not found: {slrt_bounds_csv_path}")
        return False, None
    
    print(f"✅ Found SLRT bounds CSV file: {slrt_bounds_csv_path}")
    print(f"✅ Found All questions metadata CSV file: {all_questions_metadata_csv_path}")
    
    return True, all_questions_metadata_csv_path, all_questions_json_path, slrt_bounds_csv_path

def start_server(all_questions_metadata_csv_path, all_questions_json_path, slrt_bounds_csv_path):
    """Start the Flask server with configuration."""
    print("🚀 Starting annotation server...")
    try:
        from server import app, configure
        
        # Configure the server with CSV path
        if not configure(
            all_questions_metadata_csv_path=all_questions_metadata_csv_path,
            all_questions_json_path=all_questions_json_path,
            slrt_bounds_csv_path=slrt_bounds_csv_path,
            expert_dec_column=DATA_CONFIG['expert_dec_column'],
            unannotated_value=DATA_CONFIG['unannotated_value'],
            match_value=DATA_CONFIG['match_value'],
            close_match_value=DATA_CONFIG['close_match_value'],
            vague_match_value=DATA_CONFIG['vague_match_value'],
            no_match_value=DATA_CONFIG['no_match_value']
        ):
            print("❌ Failed to load CSV file")
            return False
        
        print("\n" + "=" * 60)
        print("🌐 Server is running at: http://localhost:5001")
        print("📊 Health check: http://localhost:5001/health")
        print("📁 CSV will be updated in real-time")
        print("=" * 60)
        print("Press Ctrl+C to stop the server")
        print("=" * 60 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5001)
        
    except ImportError as e:
        error_msg = str(e)
        print(f"❌ Error importing server: {error_msg}")
        
        # Check for common numpy/pandas compatibility issues
        if "numpy.dtype size changed" in error_msg or "binary incompatibility" in error_msg:
            print("\n" + "=" * 60)
            print("🔧 BINARY INCOMPATIBILITY DETECTED")
            print("=" * 60)
            print("This is usually caused by incompatible numpy/pandas versions.")
            print("\nTo fix this, run:")
            print("  pip uninstall -y numpy pandas")
            print("  pip install numpy pandas")
            print("\nOr force reinstall:")
            print("  pip install --upgrade --force-reinstall numpy pandas")
            print("=" * 60)
        else:
            print("Make sure all dependencies are installed:")
            print("  pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("🏥 Dynamic Annotation Tool - Server Startup")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('server.py'):
        print("❌ Error: server.py not found!")
        print("Please run this script from the webpage directory.")
        sys.exit(1)
    
    # Install requirements - commented out because it is already installed
    # if not install_requirements():
    #     print("❌ Failed to install dependencies. Please install manually:")
    #     print("   pip install -r requirements.txt")
    #     sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🔧 Validating All questions CSV file...")
    print("=" * 60)
    
    # Validate CSV path
    valid, all_questions_metadata_csv_path, all_questions_json_path, slrt_bounds_csv_path = validate_data_paths()
    if not valid:
        print("\n❌ All questions CSV validation failed!")
        print("Please update the DATA_CONFIG in start_server.py with correct path.")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("📋 Configuration:")
    print(f"   All questions CSV: {os.path.basename(all_questions_metadata_csv_path)}")
    print(f"   All questions JSON: {os.path.basename(all_questions_json_path)}")
    print(f"   SLRT bounds CSV: {os.path.basename(slrt_bounds_csv_path)}")
    print(f"   Column: {DATA_CONFIG['expert_dec_column']}")
    print(f"   Values: Match={DATA_CONFIG['match_value']}, Close-match={DATA_CONFIG['close_match_value']}, Vague-match={DATA_CONFIG['vague_match_value']}, No-match={DATA_CONFIG['no_match_value']}")
    print("=" * 60)
    
    # Start the server
    start_server(all_questions_metadata_csv_path, all_questions_json_path, slrt_bounds_csv_path)
