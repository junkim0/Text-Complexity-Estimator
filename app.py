import os
import json
import torch
import numpy as np
from flask import Flask, render_template, request, jsonify
from transformers import AutoTokenizer
from train import TextComplexityModel
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Global variables for model
model = None
tokenizer = None
config = None
device = None

def load_model():
    """Load the trained model"""
    global model, tokenizer, config, device
    
    model_dir = 'model/best'
    
    if not os.path.exists(model_dir):
        return False, "Model not found. Please train the model first."
    
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Initialize model
        model = TextComplexityModel(config['model_name'])
        model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pth'), map_location=device))
        model.to(device)
        model.eval()
        
        return True, "Model loaded successfully"
    
    except Exception as e:
        return False, f"Error loading model: {str(e)}"

def predict_complexity(text):
    """Predict complexity score for text"""
    global model, tokenizer, config, device
    
    if model is None:
        return None, "Model not loaded"
    
    try:
        # Tokenize the text
        encoding = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=config['max_length'],
            return_tensors='pt'
        )
        
        # Move to device
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        
        # Make prediction
        with torch.no_grad():
            _, prediction = model(input_ids, attention_mask)
            prediction = prediction.cpu().numpy()
        
        return prediction, None
    
    except Exception as e:
        return None, f"Error making prediction: {str(e)}"

def interpret_complexity(score):
    """Interpret the complexity score"""
    if score < 0.3:
        level = "Beginner"
        description = "Very simple text, suitable for early readers"
        color = "#28a745"  # Green
    elif score < 0.5:
        level = "Elementary"
        description = "Simple text with basic vocabulary and structure"
        color = "#17a2b8"  # Blue
    elif score < 0.7:
        level = "Intermediate"
        description = "Moderate complexity with varied vocabulary"
        color = "#ffc107"  # Yellow
    elif score < 0.85:
        level = "Advanced"
        description = "Complex text with sophisticated language"
        color = "#fd7e14"  # Orange
    else:
        level = "Expert"
        description = "Very complex text requiring advanced reading skills"
        color = "#dc3545"  # Red
    
    return level, description, color

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for text complexity prediction"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'No text provided'
            })
        
        # Make prediction
        prediction, error = predict_complexity(text)
        
        if error:
            return jsonify({
                'success': False,
                'error': error
            })
        
        # Interpret result
        level, description, color = interpret_complexity(prediction)
        
        return jsonify({
            'success': True,
            'text': text,
            'score': float(prediction),
            'level': level,
            'description': description,
            'color': color,
            'timestamp': datetime.now().isoformat()
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        })

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """API endpoint for batch text complexity prediction"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({
                'success': False,
                'error': 'No texts provided'
            })
        
        results = []
        
        for text in texts:
            text = text.strip()
            if not text:
                continue
            
            prediction, error = predict_complexity(text)
            
            if error:
                results.append({
                    'text': text,
                    'success': False,
                    'error': error
                })
            else:
                level, description, color = interpret_complexity(prediction)
                results.append({
                    'text': text,
                    'success': True,
                    'score': float(prediction),
                    'level': level,
                    'description': description,
                    'color': color
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total': len(texts),
            'successful': len([r for r in results if r['success']])
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        })

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload for batch prediction"""
    try:
        if 'file' not in request.files:
            return jsonify({
                'success': False,
                'error': 'No file uploaded'
            })
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'error': 'No file selected'
            })
        
        # Read CSV file
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Error reading CSV file: {str(e)}'
            })
        
        if 'text' not in df.columns:
            return jsonify({
                'success': False,
                'error': 'CSV file must contain a "text" column'
            })
        
        texts = df['text'].tolist()
        
        # Make predictions
        results = []
        for text in texts:
            text = str(text).strip()
            if not text:
                continue
            
            prediction, error = predict_complexity(text)
            
            if error:
                results.append({
                    'text': text,
                    'success': False,
                    'error': error
                })
            else:
                level, description, color = interpret_complexity(prediction)
                results.append({
                    'text': text,
                    'success': True,
                    'score': float(prediction),
                    'level': level,
                    'description': description,
                    'color': color
                })
        
        return jsonify({
            'success': True,
            'results': results,
            'total': len(texts),
            'successful': len([r for r in results if r['success']])
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Server error: {str(e)}'
        })

@app.route('/health')
def health():
    """Health check endpoint"""
    model_loaded = model is not None
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'device': str(device) if device else None
    })

if __name__ == '__main__':
    # Load model on startup
    print("Loading model...")
    success, message = load_model()
    print(message)
    
    if success:
        print("Starting Flask app...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please train the model first.")
        print("You can still run the app, but predictions will fail.")
        app.run(debug=True, host='0.0.0.0', port=5000) 