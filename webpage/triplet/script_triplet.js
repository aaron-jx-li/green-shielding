// Global variable to store current question data
let currentQuestion = null;

// Simple script to load and display triplet data
async function loadTriplet() {
    console.log('🔄 Loading triplet data...');
    try {
        const response = await fetch('/get_triplet');
        console.log('📡 Response status:', response.status);
        
        const data = await response.json();
        console.log('📦 Response data:', data);
        
        if (!data.success) {
            console.error('❌ Error loading triplet:', data.error);
            const errorMsg = data.error || 'Unknown error';
            document.getElementById('question_og').textContent = `Error: ${errorMsg}`;
            document.getElementById('response1').textContent = `Error: ${errorMsg}`;
            document.getElementById('response2').textContent = `Error: ${errorMsg}`;
            document.getElementById('response3').textContent = `Error: ${errorMsg}`;
            return;
        }
        
        // Check if all questions are completed
        if (data.completed) {
            if (data.stats) {
                updateProgress(data.stats);
            }
            showCompletion(data.stats);
            return;
        }
        
        // Store current question data
        currentQuestion = data;
        
        console.log('✅ Data loaded successfully');
        
        // Update progress bar if stats are available
        if (data.stats) {
            updateProgress(data.stats);
        }
        
        // Display the question
        document.getElementById('question_og').textContent = data.question_og;
        
        // Display the three responses
        document.getElementById('header_response1').textContent = 'Response 1';
        document.getElementById('response1').textContent = data.response1_value;
        
        document.getElementById('header_response2').textContent = 'Response 2';
        document.getElementById('response2').textContent = data.response2_value;
        
        document.getElementById('header_response3').textContent = 'Response 3';
        document.getElementById('response3').textContent = data.response3_value;
        
        // Collapse all sections
        collapseAllSections();
        
        // Clear radio buttons and checkboxes
        document.querySelectorAll('input[name="annotation"]').forEach(radio => radio.checked = false);
        document.querySelectorAll('input[name="correctness"]').forEach(checkbox => checkbox.checked = false);
        
        // Disable next button until selection is made
        document.getElementById('nextBtn').disabled = true;
        
    } catch (error) {
        console.error('❌ Error loading triplet:', error);
        const errorMsg = `Error: ${error.message || 'Server not available'}`;
        document.getElementById('question_og').textContent = errorMsg;
        document.getElementById('response1').textContent = errorMsg;
        document.getElementById('response2').textContent = errorMsg;
        document.getElementById('response3').textContent = errorMsg;
    }
}

function validateContentVsCorrectness(showAlert = true) {
    const selectedRadio = document.querySelector('input[name="annotation"]:checked');
    if (!selectedRadio) return false;

    // ✅ Normalize (prevents "resp1_outlier " / casing issues)
    const key = (selectedRadio.value || "").trim();

    const correctness = {
        1: document.getElementById('resp1CorrectCheckbox').checked,
        2: document.getElementById('resp2CorrectCheckbox').checked,
        3: document.getElementById('resp3CorrectCheckbox').checked,
    };

    // ✅ Mapping table (more reliable than switch)
    const requiredMap = {
        resp1_outlier: [2, 3],
        resp2_outlier: [1, 3],
        resp3_outlier: [1, 2],
        all_resp_match: [1, 2, 3],
        none_resp_match: null, // no constraint
    };

    // 🔎 Debug (temporarily keep this to confirm what key is)
    console.log("validateContentVsCorrectness:", { value: selectedRadio.value, key, correctness });

    // If key is unknown, FAIL CLOSED (don’t allow next)
    if (!(key in requiredMap)) {
        if (showAlert) {
            alert(`Unknown annotation option value: "${selectedRadio.value}". Check your radio input values.`);
        }
        return false;
    }

    const required = requiredMap[key];
    if (required === null) return true; // none_resp_match

    const missing = required.filter(r => !correctness[r]);

    if (missing.length > 0) {
        if (showAlert) {
            alert(
                `You indicated Responses ${required.join(" & ")} match in diagnostic content.\n\n` +
                `Please mark ALL matching responses as correct.\n\n` +
                `Missing: Response ${missing.join(", ")}`
            );
        }
        return false;
    }

    return true;
}


async function saveAndNext() {
    const selectedRadio = document.querySelector('input[name="annotation"]:checked');
    
    if (!selectedRadio) {
        alert('Please select an annotation option before proceeding.');
        return;
    }

    if (!validateContentVsCorrectness()) {
        return;
    }

    
    if (!currentQuestion) {
        alert('No question loaded.');
        return;
    }
    
    const annotation = selectedRadio.value;
    const csvIndex = currentQuestion.csv_index;
    
    // Get checkbox values for correctness
    const resp1Correct = document.getElementById('resp1CorrectCheckbox').checked;
    const resp2Correct = document.getElementById('resp2CorrectCheckbox').checked;
    const resp3Correct = document.getElementById('resp3CorrectCheckbox').checked;
    
    try {
        // Disable button while saving
        const nextBtn = document.getElementById('nextBtn');
        nextBtn.disabled = true;
        nextBtn.textContent = 'Saving...';
        
        const response = await fetch('/save_annotation', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                annotation: annotation,
                csv_index: csvIndex,
                resp1_correct: resp1Correct ? 1 : 0,
                resp2_correct: resp2Correct ? 1 : 0,
                resp3_correct: resp3Correct ? 1 : 0
            })
        });
        
        const data = await response.json();
        
        if (!data.success) {
            alert('Failed to save: ' + (data.error || 'Unknown error'));
            nextBtn.disabled = false;
            nextBtn.textContent = 'Next Question';
            return;
        }
        
        // Check if all questions are completed
        if (data.completed) {
            showCompletion(data.stats);
            return;
        }
        
        // Update progress if stats are available
        if (data.stats) {
            updateProgress(data.stats);
        }
        
        // Load next question
        await loadTriplet();
        nextBtn.textContent = 'Next Question';
        
    } catch (error) {
        console.error('Error saving annotation:', error);
        alert('❌ Error saving annotation. Server may be unavailable.');
        
        // Re-enable button
        const nextBtn = document.getElementById('nextBtn');
        nextBtn.disabled = false;
        nextBtn.textContent = 'Next Question';
    }
}

function updateProgress(stats) {
    if (!stats) return;
    
    const progressPercentage = (stats.annotated / stats.total) * 100;
    document.getElementById('progressFill').style.width = `${progressPercentage}%`;
    document.getElementById('progressText').textContent = `${stats.annotated} / ${stats.total}`;
}

function showCompletion(stats) {
    document.getElementById('annotationCard').style.display = 'none';
    document.getElementById('summarySection').style.display = 'block';
    
    // Update progress to 100%
    if (stats) {
        updateProgress(stats);
    }
}

function handleAnnotationChange() {
    const selectedRadio = document.querySelector('input[name="annotation"]:checked');
    const nextBtn = document.getElementById('nextBtn');

    if (!selectedRadio) {
        nextBtn.disabled = true;
        return;
    }

    nextBtn.disabled = !validateContentVsCorrectness(false);
}


function collapseAllSections() {
    const sections = ['question_og', 'response1', 'response2', 'response3'];
    
    sections.forEach(sectionId => {
        const content = document.getElementById(`content_${sectionId}`);
        const icon = document.getElementById(`icon_${sectionId}`);
        
        if (content && icon) {
            content.classList.add('collapsed');
            icon.textContent = '▼';
        }
    });
}

function toggleSection(sectionId) {
    const content = document.getElementById(`content_${sectionId}`);
    const icon = document.getElementById(`icon_${sectionId}`);
    
    content.classList.toggle('collapsed');
    
    // Update icon
    if (content.classList.contains('collapsed')) {
        icon.textContent = '▼';
    } else {
        icon.textContent = '▲';
    }
}

// Load data when page loads
document.addEventListener('DOMContentLoaded', () => {
    loadTriplet();
    
    // Bind events
    document.getElementById('nextBtn').addEventListener('click', saveAndNext);
    
    // Radio button change events
    document.querySelectorAll('input[name="annotation"]').forEach(radio => {
        radio.addEventListener('change', handleAnnotationChange);
    });

    document.querySelectorAll('input[name="correctness"]').forEach(cb => {
    cb.addEventListener('change', handleAnnotationChange);
    });

});
