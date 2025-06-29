import os
import json
import torch
import numpy as np
from flask import Flask, request, jsonify, render_template, send_file
from transformers import AutoTokenizer
import pandas as pd
from train import TextComplexityModel
import io
import base64

app = Flask(__name__)

# Global variables for model
model = None
tokenizer = None
config = None

def load_model():
    """Load the trained model and configuration"""
    global model, tokenizer, config
    
    model_dir = 'model/best'
    
    # Check if model files exist
    if not os.path.exists(os.path.join(model_dir, 'best_model.pth')):
        return False, "Model files not found. Please train the model first."
    
    try:
        # Load configuration
        with open(os.path.join(model_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Initialize model
        model = TextComplexityModel(config['model_name'])
        model.load_state_dict(torch.load(os.path.join(model_dir, 'best_model.pth'), map_location='cpu'))
        model.eval()
        
        return True, "Model loaded successfully"
    except Exception as e:
        return False, f"Error loading model: {str(e)}"

def predict_complexity(text):
    """Predict complexity score for a single text"""
    if model is None or tokenizer is None:
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
        
        # Make prediction
        with torch.no_grad():
            _, prediction = model(encoding['input_ids'], encoding['attention_mask'])
            prediction = prediction.cpu().numpy()
        
        return prediction, None
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

def interpret_complexity(score):
    """Interpret the complexity score"""
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

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Single text prediction endpoint"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'success': False, 'error': 'No text provided'})
        
        # Make prediction
        prediction, error = predict_complexity(text)
        if error:
            return jsonify({'success': False, 'error': error})
        
        # Interpret result
        level, description, color = interpret_complexity(prediction)
        
        return jsonify({
            'success': True,
            'score': float(prediction),
            'level': level,
            'description': description,
            'color': color,
            'progress': int(prediction * 100)
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        data = request.get_json()
        texts = data.get('texts', [])
        
        if not texts:
            return jsonify({'success': False, 'error': 'No texts provided'})
        
        results = []
        successful = 0
        
        for text in texts:
            text = text.strip()
            if not text:
                continue
            
            prediction, error = predict_complexity(text)
            if error:
                results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'error': error
                })
            else:
                level, description, color = interpret_complexity(prediction)
                results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'score': float(prediction),
                    'level': level,
                    'description': description,
                    'color': color,
                    'progress': int(prediction * 100)
                })
                successful += 1
        
        return jsonify({
            'success': True,
            'results': results,
            'total': len(texts),
            'successful': successful
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})

@app.route('/upload', methods=['POST'])
def upload():
    """File upload endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'})
        
        if not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'error': 'Please upload a CSV file'})
        
        # Read CSV file
        try:
            df = pd.read_csv(file)
        except Exception as e:
            return jsonify({'success': False, 'error': f'Error reading CSV file: {str(e)}'})
        
        if 'text' not in df.columns:
            return jsonify({'success': False, 'error': 'CSV file must contain a "text" column'})
        
        texts = df['text'].dropna().tolist()
        if not texts:
            return jsonify({'success': False, 'error': 'No valid texts found in CSV file'})
        
        # Process texts
        results = []
        successful = 0
        
        for text in texts:
            text = str(text).strip()
            if not text:
                continue
            
            prediction, error = predict_complexity(text)
            if error:
                results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'error': error
                })
            else:
                level, description, color = interpret_complexity(prediction)
                results.append({
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'score': float(prediction),
                    'level': level,
                    'description': description,
                    'color': color,
                    'progress': int(prediction * 100)
                })
                successful += 1
        
        return jsonify({
            'success': True,
            'results': results,
            'total': len(texts),
            'successful': successful
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': f'Server error: {str(e)}'})

@app.route('/health')
def health():
    """Health check endpoint"""
    model_loaded = model is not None and tokenizer is not None
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'version': '2.0'
    })

# Load model on startup
@app.before_first_request
def initialize():
    """Initialize the model before first request"""
    success, message = load_model()
    if not success:
        print(f"Warning: {message}")

# For Vercel deployment
if __name__ == '__main__':
    # Load model
    success, message = load_model()
    if success:
        print("Model loaded successfully!")
    else:
        print(f"Warning: {message}")
    
    # Run app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 