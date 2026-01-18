// Dynamic annotation app with server-side question selection
class AnnotationApp {
    constructor() {
        this.currentQuestion = null;
        this.stats = null;
        this.isCompleted = false;
        this.userId = this.extractUserIdFromURL();
        
        // Only initialize if we have a valid userId (not redirected to login)
        if (this.userId) {
            this.initializeApp();
        }
    }

    extractUserIdFromURL() {
        // Get username from localStorage (set during login)
        const username = localStorage.getItem('annotation_username');
        if (!username) {
            // If no username found, redirect to login page
            window.location.href = 'login.html';
            return null;
        }
        // Use username as user_id (backend will handle it)
        return username;
    }

    async initializeApp() {
        this.updateUserIndicator();
        this.bindEvents();
        await this.loadNextQuestion();
    }

    updateUserIndicator() {
        // Update the user indicator in the header
        const userIndicator = document.getElementById('userIndicator');
        if (userIndicator && this.userId) {
            userIndicator.textContent = this.userId;
        }
    }

    bindEvents() {
        document.getElementById('nextBtn').addEventListener('click', () => this.saveAndNext());
        
        // Radio button change events
        document.querySelectorAll('input[name="annotation"]').forEach(radio => {
            radio.addEventListener('change', () => this.handleAnnotationChange());
        });

        // Question collapse/expand toggle
        document.getElementById('questionText').addEventListener('click', () => this.toggleQuestion());

        // Logout button
        const logoutBtn = document.getElementById('logoutBtn');
        if (logoutBtn) {
            logoutBtn.addEventListener('click', () => {
                if (confirm('Are you sure you want to logout? You will need to login again to continue.')) {
                    localStorage.removeItem('annotation_username');
                    window.location.href = 'login.html';
                }
            });
        }
    }

    toggleQuestion() {
        const questionBox = document.getElementById('questionText');
        questionBox.classList.toggle('collapsed');
    }

    extractLastSentence(text) {
        // Match sentences ending with . ! or ?
        const sentences = text.match(/[^.!?]+[.!?]+/g);
        if (!sentences || sentences.length === 0) {
            return { lastSentence: text.trim(), restOfText: '' };
        }
        
        const lastSentence = sentences[sentences.length - 1].trim();
        const restOfText = sentences.slice(0, -1).join(' ').trim();
        
        return { lastSentence, restOfText };
    }

    async loadNextQuestion() {
        try {
            const response = await fetch(`/get_next_question?user=${this.userId}`);
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

        // Update question with collapsible functionality
        const questionText = this.currentQuestion.question;
        const { lastSentence, restOfText } = this.extractLastSentence(questionText);
        
        // Set preview to placeholder message when collapsed
        document.getElementById('questionPreview').textContent = 'Click on blue arrow to open this tab and see the patient input';
        
        // Set full text with last sentence bolded
        const questionFull = document.getElementById('questionFull');
        questionFull.innerHTML = '';
        
        if (restOfText) {
            const restSpan = document.createElement('span');
            restSpan.textContent = restOfText + ' ';
            questionFull.appendChild(restSpan);
        }
        
        const lastSentenceSpan = document.createElement('span');
        lastSentenceSpan.className = 'last-sentence';
        lastSentenceSpan.textContent = lastSentence;
        questionFull.appendChild(lastSentenceSpan);
        
        // Reset to collapsed state
        document.getElementById('questionText').classList.add('collapsed');
        
        // Update other content
        document.getElementById('modelResponse').textContent = this.currentQuestion.default_response;
        document.getElementById('groundTruth').textContent = this.currentQuestion.truth;

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
            
            const response = await fetch('/save_and_next', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    annotation: annotation,
                    csv_index: csvIndex,
                    comment: comment,
                    user_id: this.userId
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
        document.getElementById('matchCount').textContent = stats.matches || 0;
        document.getElementById('closeMatchCount').textContent = stats.close_matches || 0;
        document.getElementById('vagueMatchCount').textContent = stats.vague_matches || 0;
        document.getElementById('noMatchCount').textContent = stats.no_matches || 0;
        
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
window.annotationApp = null;

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.annotationApp = new AnnotationApp();
});

// Keyboard navigation
document.addEventListener('keydown', (e) => {
    const app = window.annotationApp;
    if (!app) return;
    
    if (e.key === 'ArrowRight' || e.key === 'Enter') {
        const nextBtn = document.getElementById('nextBtn');
        if (!nextBtn.disabled) {
            nextBtn.click();
        }
    } else if (e.key === '1') {
        document.getElementById('matchRadio').click();
    } else if (e.key === '2') {
        document.getElementById('closeMatchRadio').click();
    } else if (e.key === '3') {
        document.getElementById('vagueMatchRadio').click();
    } else if (e.key === '4') {
        document.getElementById('noMatchRadio').click();
    }
});