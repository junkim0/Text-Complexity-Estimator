from flask import Flask, request, jsonify
import re

app = Flask(__name__)

@app.route('/')
def home():
    return '''<!DOCTYPE html>
<html>
<head>
    <title>Text Complexity Estimator</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
        textarea { width: 100%; height: 100px; padding: 10px; margin: 10px 0; }
        button { padding: 10px 20px; margin: 5px; background: #007bff; color: white; border: none; border-radius: 5px; }
        .result { margin-top: 20px; padding: 15px; background: #f8f9fa; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>Text Complexity Estimator</h1>
    <textarea id="text" placeholder="Enter text to analyze...">To be, or not to be, that is the question</textarea><br>
    <button onclick="analyze()">Analyze</button>
    
    <div id="result" style="display:none;" class="result">
        <div id="output"></div>
    </div>
    
    <script>
    async function analyze() {
        const text = document.getElementById('text').value.trim();
        if (!text) { alert('Enter some text'); return; }
        
        try {
            const res = await fetch('/predict', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: text})
            });
            const data = await res.json();
            
            if (data.success) {
                document.getElementById('output').innerHTML = 
                    '<strong>Score:</strong> ' + data.score.toFixed(3) + '<br>' +
                    '<strong>Level:</strong> ' + data.level + '<br>' +
                    '<strong>Description:</strong> ' + data.description;
                document.getElementById('result').style.display = 'block';
            } else {
                alert('Error: ' + data.error);
            }
        } catch (e) {
            alert('Error: ' + e.message);
        }
    }
    </script>
</body>
</html>'''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'success': False, 'error': 'No text provided'})
        
        text = data['text'].strip()
        if not text:
            return jsonify({'success': False, 'error': 'Empty text'})
        
        words = text.split()
        if not words:
            return jsonify({'success': False, 'error': 'No words found'})
        
        # Simple calculation
        avg_word_len = sum(len(w) for w in words) / len(words)
        sentences = len([s for s in re.split(r'[.!?]+', text) if s.strip()])
        avg_sent_len = len(words) / max(sentences, 1)
        archaic = len(re.findall(r'\b(thou|thee|thy|art|doth|hath)\b', text.lower()))
        
        score = 0.2 + min(avg_word_len / 20, 0.3) + min(avg_sent_len / 30, 0.2) + min(archaic / len(words) * 2, 0.3)
        score = min(max(score, 0.1), 1.0)
        
        if score < 0.3:
            level, desc = "Beginner", "Simple text"
        elif score < 0.5:
            level, desc = "Elementary", "Basic vocabulary"
        elif score < 0.7:
            level, desc = "Intermediate", "Moderate complexity"
        elif score < 0.85:
            level, desc = "Advanced", "Complex language"
        else:
            level, desc = "Expert", "Very complex text"
        
        return jsonify({
            'success': True,
            'score': score,
            'level': level,
            'description': desc
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/health')
def health():
    return jsonify({'status': 'ok'})

if __name__ == '__main__':
    app.run(debug=True) 