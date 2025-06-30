from http.server import BaseHTTPRequestHandler
import json
import re

class handler(BaseHTTPRequestHandler):
    
    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.get_home_html().encode('utf-8'))
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'status': 'ok', 'version': 'demo'}).encode('utf-8'))
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'Not found'}).encode('utf-8'))
    
    def do_POST(self):
        if self.path == '/predict':
            try:
                content_length = int(self.headers['Content-Length'])
                post_data = self.rfile.read(content_length)
                data = json.loads(post_data.decode('utf-8'))
                
                if not data or 'text' not in data:
                    self.send_error_response(400, 'No text provided')
                    return
                
                text = data['text'].strip()
                if not text:
                    self.send_error_response(400, 'Empty text')
                    return
                
                result = self.analyze_text(text)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result).encode('utf-8'))
                
            except Exception as e:
                self.send_error_response(500, str(e))
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps({'error': 'Not found'}).encode('utf-8'))
    
    def send_error_response(self, code, message):
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'success': False, 'error': message}).encode('utf-8'))
    
    def analyze_text(self, text):
        words = text.split()
        if not words:
            return {'success': False, 'error': 'No words found'}
        
        # Simple complexity calculation
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
        
        return {
            'success': True,
            'score': score,
            'level': level,
            'description': desc
        }
    
    def get_home_html(self):
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