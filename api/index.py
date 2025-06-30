from flask import Flask, request, jsonify
import re

# Create Flask app instance
app = Flask(__name__)

@app.route('/')
def home():
    return '''<!DOCTYPE html>
<html>
<head>
    <title>Text Complexity Estimator - Demo</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; background: #f0f2f5; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .demo-badge { background: #ff4757; color: white; padding: 5px 15px; border-radius: 15px; font-size: 12px; display: inline-block; margin-bottom: 15px; }
        textarea { width: 100%; height: 100px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; font-size: 14px; }
        button { background: #3742fa; color: white; padding: 10px 20px; border: none; border-radius: 5px; cursor: pointer; margin: 10px 5px 0 0; }
        button:hover { background: #2f3542; }
        .result { margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; }
        .score { font-size: 20px; font-weight: bold; margin: 10px 0; }
        .level { padding: 5px 10px; border-radius: 15px; color: white; font-weight: bold; display: inline-block; }
    </style>
</head>
<body>
    <div class="container">
        <div class="demo-badge">DEMO VERSION</div>
        <h1>Text Complexity Estimator</h1>
        <p>Analyze text complexity with rule-based estimation</p>
        
        <textarea id="text" placeholder="Enter text to analyze...">To be, or not to be, that is the question</textarea><br>
        <button onclick="analyze()">Analyze</button>
        <button onclick="clearText()">Clear</button>
        
        <div id="result" style="display:none;" class="result">
            <div id="score" class="score"></div>
            <div id="level" class="level"></div>
            <p id="desc"></p>
            <small id="note" style="color:#666;"></small>
        </div>
    </div>
    
    <script>
    async function analyze() {
        const text = document.getElementById('text').value.trim();
        if (!text) { alert('Enter some text'); return; }
        
        try {
            const res = await fetch('/api/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: text})
            });
            const data = await res.json();
            
            if (data.success) {
                document.getElementById('score').textContent = 'Score: ' + data.score.toFixed(3);
                const levelEl = document.getElementById('level');
                levelEl.textContent = data.level;
                levelEl.style.backgroundColor = data.color;
                document.getElementById('desc').textContent = data.description;
                document.getElementById('note').textContent = data.note || '';
                document.getElementById('result').style.display = 'block';
            } else {
                alert('Error: ' + data.error);
            }
        } catch (e) {
            alert('Error: ' + e.message);
        }
    }
    
    function clearText() {
        document.getElementById('text').value = '';
        document.getElementById('result').style.display = 'none';
    }
    </script>
</body>
</html>'''

@app.route('/predict', methods=['POST'])
@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'success': False, 'error': 'No text provided'})
        
        text = data['text'].strip()
        if not text:
            return jsonify({'success': False, 'error': 'Empty text'})
        
        # Simple complexity calculation
        words = text.split()
        if not words:
            return jsonify({'success': False, 'error': 'No words found'})
        
        # Basic metrics
        avg_word_len = sum(len(w) for w in words) / len(words)
        sentences = len([s for s in re.split(r'[.!?]+', text) if s.strip()])
        avg_sent_len = len(words) / max(sentences, 1)
        
        # Archaic words
        archaic = len(re.findall(r'\b(thou|thee|thy|art|doth|hath)\b', text.lower()))
        
        # Calculate score
        score = 0.2  # base
        score += min(avg_word_len / 20, 0.3)  # word length
        score += min(avg_sent_len / 30, 0.2)  # sentence length
        score += min(archaic / len(words) * 2, 0.3)  # archaic boost
        
        score = min(max(score, 0.1), 1.0)
        
        # Get level
        if score < 0.3:
            level, desc, color = "Beginner", "Simple text", "#28a745"
        elif score < 0.5:
            level, desc, color = "Elementary", "Basic vocabulary", "#17a2b8"
        elif score < 0.7:
            level, desc, color = "Intermediate", "Moderate complexity", "#ffc107"
        elif score < 0.85:
            level, desc, color = "Advanced", "Complex language", "#fd7e14"
        else:
            level, desc, color = "Expert", "Very complex text", "#dc3545"
        
        return jsonify({
            'success': True,
            'score': score,
            'level': level,
            'description': desc,
            'color': color,
            'note': 'Demo version with rule-based estimation'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
@app.route('/api/health')
def health():
    return jsonify({'status': 'ok', 'version': 'demo'})

# For local development
if __name__ == '__main__':
    app.run(debug=False) 