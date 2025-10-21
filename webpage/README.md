# AI Model Response Annotation Tool

A web-based interface for doctors to annotate AI model responses in medical scenarios.

## Features

- **Clean Interface**: UC Berkeley color scheme with professional medical interface design
- **Interactive Navigation**: Navigate through cases with Previous/Next buttons or keyboard shortcuts
- **Progress Tracking**: Visual progress bar showing annotation completion
- **Annotation Options**: Radio buttons to mark whether model responses match ground truth
- **Data Export**: Export annotations to CSV format for analysis
- **Responsive Design**: Works on desktop and mobile devices

## Usage

### Starting the Server (Recommended)
1. **Option A - Simple startup:**
   ```bash
   ./start_server.sh
   ```

2. **Option B - Manual startup:**
   ```bash
   pip install -r requirements.txt
   python3 server.py
   ```

3. **Option C - Python startup script:**
   ```bash
   python3 start_server.py
   ```

4. Open your browser and go to: **http://localhost:5000**

### Alternative: Direct File Opening
1. Open `index.html` in a web browser (limited functionality)
2. The interface will automatically load the data from `data.json`

### Annotating Cases
1. Read the **Question** (medical scenario)
2. Review the **Model Response** (AI-generated answer)
3. Compare with **Ground Truth** (correct answer)
4. Select either:
   - "Model matches ground truth" 
   - "Model does not match ground truth"
5. Use Previous/Next buttons to navigate between cases

### Keyboard Shortcuts
- `←` (Left Arrow): Previous case
- `→` (Right Arrow): Next case
- `1`: Select "Model matches ground truth"
- `2`: Select "Model does not match ground truth"

### Exporting Results
- Click "Save Annotations" to save progress
- Click "Export Results" to download a CSV file with all annotations

## Data Format

The website expects a JSON file (`data.json`) with the following structure:
```json
[
  {
    "question": "Medical question text...",
    "default_response": "AI model response...",
    "truth": "Ground truth answer...",
    "judge_wq5_dec": "True/False"
  }
]
```

## File Structure
```
webpage/
├── index.html          # Main webpage
├── styles.css           # UC Berkeley themed styling
├── script.js           # Interactive functionality
├── data.json           # Medical cases data
├── convert_data.py     # Script to convert CSV to JSON
├── server.py           # Flask server for saving annotations
├── start_server.py     # Python startup script
├── start_server.sh     # Bash startup script
├── requirements.txt    # Python dependencies
├── annotations/        # Directory for saved annotation files
└── README.md           # This file
```

## Saving Annotations

### Server Mode (Recommended)
- **Location**: `./annotations/` directory
- **Formats**: Both JSON and CSV files
- **Files created**:
  - `annotations_YYYYMMDD_HHMMSS.json` (timestamped)
  - `annotations_YYYYMMDD_HHMMSS.csv` (timestamped)
  - `latest_annotations.json` (always latest)
  - `latest_annotations.csv` (always latest)

### Browser Mode (Fallback)
- **Location**: Browser local storage + Downloads folder
- **Formats**: CSV download only
- **Persistence**: Survives browser restarts

## Converting New Data

To use different CSV data:
1. Update the path in `convert_data.py`
2. Run: `python convert_data.py`
3. Refresh the webpage

## UC Berkeley Design Elements

- **Primary Colors**: Berkeley Blue (#003262), Berkeley Gold (#FDB515)
- **Accent Colors**: Berkeley Orange (#F4801A), Berkeley Light Blue (#00A598)
- **Typography**: Clean, professional fonts with good readability
- **Layout**: Card-based design with clear visual hierarchy
