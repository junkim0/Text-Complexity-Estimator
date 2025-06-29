import os
import json
import numpy as np
from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import re
import io

# Initialize Flask app with simpler configuration
app = Flask(__name__)

# Simple HTML template embedded in the code to avoid path issues
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Text Complexity Estimator - Demo</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            background: white;
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .demo-badge {
            background: #ff6b6b;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            display: inline-block;
            margin-bottom: 10px;
        }
        textarea {
            width: 100%;
            height: 120px;
            padding: 15px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 16px;
            resize: vertical;
        }
        .btn {
            background: #667eea;
            color: white;
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            font-size: 16px;
            cursor: pointer;
            margin: 10px 5px;
        }
        .btn:hover {
            background: #5a6fd8;
        }
        .result {
            margin-top: 20px;
            padding: 20px;
            border-radius: 8px;
            background: #f8f9fa;
        }
        .score-display {
            font-size: 24px;
            font-weight: bold;
            margin: 10px 0;
        }
        .complexity-level {
            padding: 8px 16px;
            border-radius: 20px;
            color: white;
            font-weight: bold;
            display: inline-block;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="demo-badge">DEMO VERSION</div>
            <h1>Text Complexity Estimator</h1>
            <p>Analyze text complexity using AI-powered analysis</p>
        </div>
        
        <div>
            <textarea id="textInput" placeholder="Enter your text here to analyze its complexity...">To be, or not to be, that is the question</textarea>
            <br>
            <button class="btn" onclick="analyzeText()">Analyze Complexity</button>
            <button class="btn" onclick="clearText()">Clear</button>
        </div>
        
        <div id="result" class="result" style="display: none;">
            <h3>Analysis Result</h3>
            <div id="scoreDisplay" class="score-display"></div>
            <div id="levelDisplay" class="complexity-level"></div>
            <p id="description"></p>
            <p><small id="note" style="color: #666;"></small></p>
        </div>
    </div>

    <script>
        async function analyzeText() {
            const text = document.getElementById('textInput').value.trim();
            if (!text) {
                alert('Please enter some text to analyze');
                return;
            }

            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ text: text })
                });

                const data = await response.json();
                
                if (data.success) {
                    document.getElementById('scoreDisplay').textContent = `Complexity Score: ${data.score.toFixed(3)}`;
                    
                    const levelEl = document.getElementById('levelDisplay');
                    levelEl.textContent = data.level;
                    levelEl.style.backgroundColor = data.color;
                    
                    document.getElementById('description').textContent = data.description;
                    document.getElementById('note').textContent = data.note || '';
                    document.getElementById('result').style.display = 'block';
                } else {
                    alert('Error: ' + data.error);
                }
            } catch (error) {
                alert('Error analyzing text: ' + error.message);
            }
        }

        function clearText() {
            document.getElementById('textInput').value = '';
            document.getElementById('result').style.display = 'none';
        }
    </script>
</body>
</html>
"""

def estimate_complexity_simple(text):
    """Simple rule-based complexity estimation for demo purposes"""
    try:
        if not text or len(text.strip()) == 0:
            return 0.1
        
        # Calculate various complexity metrics
        words = text.split()
        sentences = len(re.split(r'[.!?]+', text))
        
        if len(words) == 0:
            return 0.1
        
        # Average word length
        avg_word_length = sum(len(word.strip('.,!?;:"()[]{}')) for word in words) / len(words)
        
        # Average sentence length
        avg_sentence_length = len(words) / max(sentences, 1)
        
        # Count complex words (more than 6 characters)
        complex_words = sum(1 for word in words if len(word.strip('.,!?;:"()[]{}')) > 6)
        complex_word_ratio = complex_words / len(words)
        
        # Count archaic/formal words (simple heuristic)
        archaic_patterns = [
            r'\b(thou|thee|thy|thine|art|doth|hath|whence|whither|wherefore|albeit|forsooth)\b',
            r'\b\w+eth\b',  # words ending in 'eth'
            r'\b\w+st\b',   # words ending in 'st' (archaic verb forms)
        ]
        
        archaic_count = 0
        for pattern in archaic_patterns:
            archaic_count += len(re.findall(pattern, text.lower()))
        
        archaic_ratio = archaic_count / len(words)
        
        # Calculate base complexity score
        complexity = 0.0
        
        # Word length factor (0-0.3)
        complexity += min(avg_word_length / 15, 0.3)
        
        # Sentence length factor (0-0.2)
        complexity += min(avg_sentence_length / 25, 0.2)
        
        # Complex words factor (0-0.3)
        complexity += complex_word_ratio * 0.3
        
        # Archaic language boost (can add up to 0.4)
        if archaic_ratio > 0:
            complexity += min(archaic_ratio * 2, 0.4)
        
        # Text length factor (longer texts tend to be more complex)
        length_factor = min(len(text) / 1000, 0.2)
        complexity += length_factor
        
        # Ensure score is between 0 and 1
        return min(max(complexity, 0.1), 1.0)
    except Exception:
        # Fallback to simple estimation if anything fails
        return 0.5

def interpret_complexity(score):
    """Interpret the complexity score"""
    try:
        if score < 0.3:
            level = "Beginner"
            description = "Very simple text, suitable for early readers"
            color = "#28a745"
        elif score < 0.5:
            level = "Elementary"
            description = "Simple text with basic vocabulary and structure"
            color = "#17a2b8"
        elif score < 0.7:
            level = "Intermediate"
            description = "Moderate complexity with varied vocabulary"
            color = "#ffc107"
        elif score < 0.85:
            level = "Advanced"
            description = "Complex text with sophisticated language"
            color = "#fd7e14"
        else:
            level = "Expert"
            description = "Very complex text requiring advanced reading skills"
            color = "#dc3545"
        
        return level, description, color
    except Exception:
        return "Unknown", "Unable to determine complexity", "#6c757d"

@app.route('/')
def index():
    """Main page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """Single text prediction endpoint"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'No JSON data provided'})
            
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'success': False, 'error': 'No text provided'})
        
        # Make prediction using simple estimator
        prediction = estimate_complexity_simple(text)
        
        # Interpret result
        level, description, color = interpret_complexity(prediction)
        
        return jsonify({
            'success': True,
            'score': float(prediction),
            'level': level,
            'description': description,
            'color': color,
            'progress': int(prediction * 100),
            'demo_mode': True,
            'note': 'Running in demo mode with rule-based complexity estimation'
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': False,
        'demo_mode': True,
        'version': '2.0-demo',
        'note': 'Running in demo mode with rule-based complexity estimation'
    })

# Export the Flask app for Vercel
# This is required for Vercel serverless deployment
if __name__ == '__main__':
    # Run app locally
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 