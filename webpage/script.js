// Data will be loaded from JSON file
let data = [];

class AnnotationApp {
    constructor() {
        this.data = data;
        this.currentIndex = 0;
        this.annotations = {};
        this.loadData();
    }

    async loadData() {
        try {
            const response = await fetch('data.json');
            this.data = await response.json();
            this.initializeApp();
        } catch (error) {
            console.error('Error loading data:', error);
            // Fallback to sample data if loading fails
            this.data = [
                {
                    question: "Sample question for testing",
                    default_response: "Sample model response",
                    truth: "Sample ground truth",
                    judge_wq5_dec: "True"
                }
            ];
            this.initializeApp();
        }
    }

    initializeApp() {
        this.bindEvents();
        this.updateDisplay();
        this.updateProgress();
    }

    bindEvents() {
        document.getElementById('prevBtn').addEventListener('click', () => this.previousCase());
        document.getElementById('nextBtn').addEventListener('click', () => this.nextCase());
        document.getElementById('saveBtn').addEventListener('click', () => this.saveAnnotations());
        document.getElementById('exportBtn').addEventListener('click', () => this.exportResults());
        
        // Radio button change events
        document.querySelectorAll('input[name="annotation"]').forEach(radio => {
            radio.addEventListener('change', () => this.handleAnnotationChange());
        });
    }

    updateDisplay() {
        const currentCase = this.data[this.currentIndex];
        if (!currentCase) return;

        // Update case number
        document.getElementById('caseNumber').textContent = this.currentIndex + 1;

        // Update content
        document.getElementById('questionText').textContent = currentCase.question;
        document.getElementById('modelResponse').textContent = currentCase.default_response;
        document.getElementById('groundTruth').textContent = currentCase.truth;

        // Update radio buttons based on existing annotation
        const annotation = this.annotations[this.currentIndex];
        if (annotation) {
            document.getElementById(annotation === 'match' ? 'matchRadio' : 'noMatchRadio').checked = true;
        } else {
            document.querySelectorAll('input[name="annotation"]').forEach(radio => radio.checked = false);
        }

        // Update button states
        this.updateButtonStates();
    }

    updateButtonStates() {
        const prevBtn = document.getElementById('prevBtn');
        const nextBtn = document.getElementById('nextBtn');
        const saveBtn = document.getElementById('saveBtn');

        prevBtn.disabled = this.currentIndex === 0;
        nextBtn.disabled = this.currentIndex === this.data.length - 1;
        
        const hasAnnotation = this.annotations[this.currentIndex];
        saveBtn.disabled = !hasAnnotation;
    }

    updateProgress() {
        const totalCases = this.data.length;
        const annotatedCases = Object.keys(this.annotations).length;
        const progressPercentage = (annotatedCases / totalCases) * 100;

        document.getElementById('progressFill').style.width = `${progressPercentage}%`;
        document.getElementById('progressText').textContent = `${annotatedCases} / ${totalCases}`;

        // Update summary if all cases are annotated
        if (annotatedCases === totalCases) {
            this.showSummary();
        }
    }

    handleAnnotationChange() {
        const selectedRadio = document.querySelector('input[name="annotation"]:checked');
        if (selectedRadio) {
            this.annotations[this.currentIndex] = selectedRadio.value;
            this.updateButtonStates();
            this.updateProgress();
        }
    }

    previousCase() {
        if (this.currentIndex > 0) {
            this.currentIndex--;
            this.updateDisplay();
            this.updateProgress();
        }
    }

    nextCase() {
        if (this.currentIndex < this.data.length - 1) {
            this.currentIndex++;
            this.updateDisplay();
            this.updateProgress();
        }
    }

    saveAnnotations() {
        // Save annotations to server only
        console.log('Annotations to be saved:', this.annotations);
        alert('Click "Export Results" to save annotations to the server files.');
    }

    showSummary() {
        const summarySection = document.getElementById('summarySection');
        const totalCases = this.data.length;
        const annotatedCases = Object.keys(this.annotations).length;
        const matchCount = Object.values(this.annotations).filter(ann => ann === 'match').length;
        const noMatchCount = Object.values(this.annotations).filter(ann => ann === 'no-match').length;

        document.getElementById('totalCases').textContent = totalCases;
        document.getElementById('annotatedCases').textContent = annotatedCases;
        document.getElementById('matchCount').textContent = matchCount;
        document.getElementById('noMatchCount').textContent = noMatchCount;

        summarySection.style.display = 'block';
        summarySection.scrollIntoView({ behavior: 'smooth' });
    }

    exportResults() {
        const results = this.data.map((caseData, index) => ({
            case_number: index + 1,
            question: caseData.question,
            model_response: caseData.default_response,
            ground_truth: caseData.truth,
            annotation: this.annotations[index] || 'not_annotated',
            original_judge_decision: caseData.judge_wq5_dec
        }));

        // Save to server only (no browser download)
        this.saveToServer(results);
    }

    async saveToServer(results) {
        try {
            const response = await fetch('/save_annotations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(results)
            });
            
            if (response.ok) {
                const result = await response.json();
                console.log('✅ Annotations saved to server:', result);
                alert(`✅ Annotations saved to server!\nFiles created: ${result.files_created.length}\nTimestamp: ${result.timestamp}\nLocation: ./annotations/`);
            } else {
                console.error('Server save failed:', response.statusText);
                alert('❌ Server save failed! Make sure the Python server is running.');
            }
        } catch (error) {
            console.log('❌ Server not available');
            alert('❌ Server not available! Please start the Python server first.\nRun: python3 server.py');
        }
    }

    // CSV conversion and download functions removed - using server-only saves
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new AnnotationApp();
});

// Keyboard navigation
document.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowLeft') {
        document.getElementById('prevBtn').click();
    } else if (e.key === 'ArrowRight') {
        document.getElementById('nextBtn').click();
    } else if (e.key === '1') {
        document.getElementById('matchRadio').click();
    } else if (e.key === '2') {
        document.getElementById('noMatchRadio').click();
    }
});
