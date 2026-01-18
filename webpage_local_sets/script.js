// Diagnosis annotation app
class DiagnosisAnnotationApp {
    constructor() {
        this.currentQuestion = null;
        this.stats = null;
        this.currentUserId = null;
        this.availableUsers = [];
        this.initializeApp();
    }

    async initializeApp() {
        this.bindEvents();
        
        // Load available users first
        await this.loadAvailableUsers();
        
        // Check for stored user_id in localStorage and validate it
        const storedUserId = localStorage.getItem('annotation_user_id');
        if (storedUserId && this.availableUsers.includes(storedUserId)) {
            await this.login(storedUserId);
        } else {
            // Clear invalid stored user_id
            if (storedUserId) {
                localStorage.removeItem('annotation_user_id');
            }
            this.showLoginScreen();
        }
    }

    bindEvents() {
        document.getElementById('nextBtn').addEventListener('click', () => this.saveAndNext());
    }

    async loadAvailableUsers() {
        try {
            const response = await fetch('/users');
            const data = await response.json();
            
            if (data.success && data.users) {
                this.availableUsers = data.users;
            } else {
                // Fallback to default users
                this.availableUsers = ['user1', 'user2', 'user3', 'user4'];
            }
        } catch (error) {
            console.error('Error loading users:', error);
            // Fallback to default users if server unavailable
            this.availableUsers = ['user1', 'user2', 'user3', 'user4'];
        }
    }

    showLoginScreen() {
        const loginSection = document.getElementById('loginSection');
        const annotationCard = document.getElementById('annotationCard');
        const progressBar = document.getElementById('progressBar');
        const userInfo = document.getElementById('userInfo');
        
        loginSection.style.display = 'block';
        annotationCard.style.display = 'none';
        progressBar.style.display = 'none';
        userInfo.style.display = 'none';
        
        // Populate user buttons
        const userButtons = document.getElementById('userButtons');
        userButtons.innerHTML = '';
        
        this.availableUsers.forEach(userId => {
            const button = document.createElement('button');
            button.className = 'btn btn-primary';
            button.textContent = userId;
            button.addEventListener('click', () => this.login(userId));
            userButtons.appendChild(button);
        });
    }

    hideLoginScreen() {
        const loginSection = document.getElementById('loginSection');
        const annotationCard = document.getElementById('annotationCard');
        const progressBar = document.getElementById('progressBar');
        const userInfo = document.getElementById('userInfo');
        
        loginSection.style.display = 'none';
        annotationCard.style.display = 'block';
        progressBar.style.display = 'block';
        userInfo.style.display = 'block';
    }

    async login(userId) {
        try {
            const response = await fetch('/login', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ user_id: userId })
            });
            
            const data = await response.json();
            
            if (!data.success) {
                // Clear invalid stored user_id if login fails
                const storedUserId = localStorage.getItem('annotation_user_id');
                if (storedUserId === userId) {
                    localStorage.removeItem('annotation_user_id');
                }
                this.showError('Login failed: ' + (data.error || 'Unknown error'));
                this.showLoginScreen();
                return;
            }
            
            this.currentUserId = userId;
            localStorage.setItem('annotation_user_id', userId);
            
            // Update user info display
            document.getElementById('userInfo').textContent = `Logged in as: ${userId}`;
            
            this.hideLoginScreen();
            await this.loadNextQuestion();
            
        } catch (error) {
            console.error('Error logging in:', error);
            // Clear stored user_id on server error
            localStorage.removeItem('annotation_user_id');
            this.showError('❌ Server not available! Please start the Python server first.\nRun: python3 server.py');
            this.showLoginScreen();
        }
    }

    async loadNextQuestion() {
        if (!this.currentUserId) {
            this.showError('No user logged in');
            return;
        }
        
        try {
            const response = await fetch(`/get_next_question?user_id=${encodeURIComponent(this.currentUserId)}`);
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
            this.showError('❌ Server not available! Please start the Python server first.\nRun: python3 server.py');
        }
    }

    updateDisplay() {
        if (!this.currentQuestion) return;

        // Update patient input
        document.getElementById('patientInput').textContent = this.currentQuestion.input || 'N/A';

        // Update Question 1: Not plausible checkboxes
        const notPlausibleGroup = document.getElementById('notPlausibleGroup');
        notPlausibleGroup.innerHTML = '';
        this.currentQuestion.plausible_set.forEach((diagnosis, idx) => {
            const label = document.createElement('label');
            label.className = 'checkbox-option';
            label.innerHTML = `
                <input type="checkbox" name="not_plausible" value="${this.escapeHtml(diagnosis)}">
                <span class="checkbox-custom"></span>
                <span class="checkbox-label">${this.escapeHtml(diagnosis)}</span>
            `;
            notPlausibleGroup.appendChild(label);
        });

        // Update Question 3: Not highly likely checkboxes
        const notHighlyLikelyGroup = document.getElementById('notHighlyLikelyGroup');
        notHighlyLikelyGroup.innerHTML = '';
        this.currentQuestion.highly_likely_set.forEach((diagnosis, idx) => {
            const label = document.createElement('label');
            label.className = 'checkbox-option';
            label.innerHTML = `
                <input type="checkbox" name="not_highly_likely" value="${this.escapeHtml(diagnosis)}">
                <span class="checkbox-custom"></span>
                <span class="checkbox-label">${this.escapeHtml(diagnosis)}</span>
            `;
            notHighlyLikelyGroup.appendChild(label);
        });

        // Clear text inputs
        document.getElementById('missingPlausible').value = '';
        document.getElementById('missingHighlyLikely').value = '';
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    updateProgress() {
        if (!this.stats) return;
        
        const progressPercentage = this.stats.total > 0 
            ? (this.stats.annotated / this.stats.total) * 100 
            : 0;
        
        document.getElementById('progressFill').style.width = `${progressPercentage}%`;
        document.getElementById('progressText').textContent = `${this.stats.annotated} / ${this.stats.total}`;
    }

    async saveAndNext() {
        if (!this.currentQuestion) {
            alert('No question loaded.');
            return;
        }
        
        // Collect answers
        const notPlausible = Array.from(document.querySelectorAll('input[name="not_plausible"]:checked'))
            .map(cb => cb.value);
        const missingPlausible = document.getElementById('missingPlausible').value.trim();
        const notHighlyLikely = Array.from(document.querySelectorAll('input[name="not_highly_likely"]:checked'))
            .map(cb => cb.value);
        const missingHighlyLikely = document.getElementById('missingHighlyLikely').value.trim();
        
        try {
            const nextBtn = document.getElementById('nextBtn');
            nextBtn.disabled = true;
            nextBtn.textContent = 'Saving...';
            
            const response = await fetch('/save_annotation', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    user_id: this.currentUserId,
                    index: this.currentQuestion.index,
                    not_plausible: notPlausible,
                    missing_plausible: missingPlausible,
                    not_highly_likely: notHighlyLikely,
                    missing_highly_likely: missingHighlyLikely
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
            
            nextBtn.disabled = false;
            nextBtn.textContent = 'Next';
            
        } catch (error) {
            console.error('Error saving annotation:', error);
            this.showError('❌ Error saving annotation. Server may be unavailable.');
            
            const nextBtn = document.getElementById('nextBtn');
            nextBtn.disabled = false;
            nextBtn.textContent = 'Next';
        }
    }

    showCompletion(stats) {
        document.getElementById('annotationCard').style.display = 'none';
        
        const summarySection = document.getElementById('summarySection');
        document.getElementById('totalCases').textContent = stats.total;
        document.getElementById('annotatedCases').textContent = stats.annotated;
        
        summarySection.style.display = 'block';
        summarySection.scrollIntoView({ behavior: 'smooth' });
        
        document.getElementById('progressFill').style.width = '100%';
        document.getElementById('progressText').textContent = `${stats.total} / ${stats.total}`;
    }

    showError(message) {
        alert(message);
    }
}

// Initialize the app when the page loads
document.addEventListener('DOMContentLoaded', () => {
    window.diagnosisAnnotationApp = new DiagnosisAnnotationApp();
});

