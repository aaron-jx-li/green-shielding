// Normalization annotation app with server-side question selection
class NormalizationAnnotationApp {
    constructor() {
        this.currentQuestion = null;
        this.stats = null;
        this.isCompleted = false;
        this.initializeApp();
    }

    async initializeApp() {
        this.bindEvents();
        await this.loadNextQuestion();
    }

    bindEvents() {
        document.getElementById('nextBtn').addEventListener('click', () => this.saveAndNext());
        
        // Radio button change events
        document.querySelectorAll('input[name="annotation"]').forEach(radio => {
            radio.addEventListener('change', () => this.handleAnnotationChange());
        });

        // Toggle mode button
        document.getElementById('toggleModeBtn').addEventListener('click', () => {
            window.location.href = '/';
        });
    }

    async loadNextQuestion() {
        try {
            const response = await fetch('/get_next_question_norm');
            const data = await response.json();
            
            if (!data.success) {
                this.showError('Failed to load question: ' + (data.error || 'Unknown error'));
                return;
            }
            
            if (data.completed) {
                this.showCompletion(data.stats);
                return;
            }
            
            this.currentQuestion = data.question;
            this.stats = data.stats;
            this.updateDisplay();
            this.updateProgress();
            
        } catch (error) {
            console.error('Error loading question:', error);
            this.showError('❌ Server not available! Please start the Python server first.\nRun: python3 start_server.py');
        }
    }

    updateDisplay() {
        if (!this.currentQuestion) return;

        // Update all content sections
        document.getElementById('originalPrompt').textContent = this.currentQuestion.question_og || 'N/A';
        document.getElementById('originalResponse').textContent = this.currentQuestion.model_response_og || 'N/A';
        document.getElementById('normalizedResponse').textContent = this.currentQuestion.model_response_conv || 'N/A';
        document.getElementById('referenceDiagnosis').textContent = this.currentQuestion.reference_diagnosis_conv || 'N/A';

        // Clear radio buttons for new question
        document.querySelectorAll('input[name="annotation"]').forEach(radio => radio.checked = false);

        // Clear comment box for new question
        document.getElementById('commentBox').value = '';

        // Update button states
        this.updateButtonStates();
    }

    updateButtonStates() {
        const nextBtn = document.getElementById('nextBtn');
        const selectedRadio = document.querySelector('input[name="annotation"]:checked');
        
        // Enable next button only if annotation is selected
        nextBtn.disabled = !selectedRadio;
        
        // Update button text
        if (this.stats && this.stats.remaining === 1 && selectedRadio) {
            nextBtn.textContent = 'Finish';
        } else {
            nextBtn.textContent = 'Next';
        }
    }

    updateProgress() {
        if (!this.stats) return;
        
        const progressPercentage = (this.stats.annotated / this.stats.total) * 100;
        
        document.getElementById('progressFill').style.width = `${progressPercentage}%`;
        document.getElementById('progressText').textContent = `${this.stats.annotated} / ${this.stats.total}`;
    }

    handleAnnotationChange() {
        this.updateButtonStates();
    }

    async saveAndNext() {
        const selectedRadio = document.querySelector('input[name="annotation"]:checked');
        
        if (!selectedRadio) {
            alert('Please select an annotation before proceeding.');
            return;
        }
        
        if (!this.currentQuestion) {
            alert('No question loaded.');
            return;
        }
        
        const annotation = selectedRadio.value;
        const csvIndex = this.currentQuestion.csv_index;
        const comment = document.getElementById('commentBox').value.trim();
        
        try {
            // Disable button while saving
            const nextBtn = document.getElementById('nextBtn');
            nextBtn.disabled = true;
            nextBtn.textContent = 'Saving...';
            
            const response = await fetch('/save_and_next_norm', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    annotation: annotation,
                    csv_index: csvIndex,
                    comment: comment
                })
            });
            
            const data = await response.json();
            
            if (!data.success) {
                this.showError('Failed to save: ' + (data.error || 'Unknown error'));
                nextBtn.disabled = false;
                nextBtn.textContent = 'Next';
                return;
            }
            
            if (data.completed) {
                this.showCompletion(data.stats);
                return;
            }
            
            // Load next question
            this.currentQuestion = data.question;
            this.stats = data.stats;
            this.updateDisplay();
            this.updateProgress();
            
        } catch (error) {
            console.error('Error saving annotation:', error);
            this.showError('❌ Error saving annotation. Server may be unavailable.');
            
            // Re-enable button
            const nextBtn = document.getElementById('nextBtn');
            nextBtn.disabled = false;
            nextBtn.textContent = 'Next';
        }
    }

    showCompletion(stats) {
        this.isCompleted = true;
        
        // Hide annotation card
        document.getElementById('annotationCard').style.display = 'none';
        
        // Show summary
        const summarySection = document.getElementById('summarySection');
        document.getElementById('totalCases').textContent = stats.total;
        document.getElementById('annotatedCases').textContent = stats.annotated;
        document.getElementById('originalCount').textContent = stats.original_selected || 0;
        document.getElementById('normalizedCount').textContent = stats.normalized_selected || 0;
        
        summarySection.style.display = 'block';
        summarySection.scrollIntoView({ behavior: 'smooth' });
        
        // Update progress to 100%
        document.getElementById('progressFill').style.width = '100%';
        document.getElementById('progressText').textContent = `${stats.total} / ${stats.total}`;
    }

    showError(message) {
        alert(message);
    }
}

// Store app instance globally for keyboard shortcuts
window.normalizationAnnotationApp = null;

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.normalizationAnnotationApp = new NormalizationAnnotationApp();
});

// Keyboard navigation
document.addEventListener('keydown', (e) => {
    const app = window.normalizationAnnotationApp;
    if (!app) return;
    
    if (e.key === 'ArrowRight' || e.key === 'Enter') {
        const nextBtn = document.getElementById('nextBtn');
        if (!nextBtn.disabled) {
            nextBtn.click();
        }
    } else if (e.key === '1') {
        document.getElementById('originalRadio').click();
    } else if (e.key === '2') {
        document.getElementById('normalizedRadio').click();
    }
});

