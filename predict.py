import os
import json
import argparse
import torch
import numpy as np
from transformers import AutoTokenizer
import pandas as pd
from train import TextComplexityModel

def load_trained_model(model_dir, device):
    """Load the trained model and configuration"""
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
    
    return model, tokenizer, config

def predict_complexity(text, model, tokenizer, config, device):
    """Predict complexity score for a single text"""
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
    
    return prediction

def interpret_complexity(score):
    """Interpret the complexity score"""
    if score < 0.3:
        level = "Beginner"
        description = "Very simple text, suitable for early readers"
    elif score < 0.5:
        level = "Elementary"
        description = "Simple text with basic vocabulary and structure"
    elif score < 0.7:
        level = "Intermediate"
        description = "Moderate complexity with varied vocabulary"
    elif score < 0.85:
        level = "Advanced"
        description = "Complex text with sophisticated language"
    else:
        level = "Expert"
        description = "Very complex text requiring advanced reading skills"
    
    return level, description

def predict_batch(texts, model, tokenizer, config, device):
    """Predict complexity scores for multiple texts"""
    predictions = []
    
    for text in texts:
        pred = predict_complexity(text, model, tokenizer, config, device)
        predictions.append(pred)
    
    return np.array(predictions)

def main():
    parser = argparse.ArgumentParser(description='Predict Text Complexity')
    parser.add_argument('--model_dir', type=str, default='model/best',
                       help='Directory containing the trained model')
    parser.add_argument('--text', type=str, default=None,
                       help='Text to predict complexity for')
    parser.add_argument('--input_file', type=str, default=None,
                       help='CSV file containing texts to predict (should have a "text" column)')
    parser.add_argument('--output_file', type=str, default=None,
                       help='Output file to save predictions (CSV format)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print detailed information')
    
    args = parser.parse_args()
    
    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory '{args.model_dir}' not found!")
        print("Please train the model first using train.py")
        return
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load trained model
    print("Loading trained model...")
    model, tokenizer, config = load_trained_model(args.model_dir, device)
    print("Model loaded successfully!")
    
    if args.text:
        # Single text prediction
        print(f"\nPredicting complexity for: '{args.text}'")
        prediction = predict_complexity(args.text, model, tokenizer, config, device)
        level, description = interpret_complexity(prediction)
        
        print("\n" + "="*60)
        print("PREDICTION RESULTS")
        print("="*60)
        print(f"Text: {args.text}")
        print(f"Complexity Score: {prediction:.4f}")
        print(f"Level: {level}")
        print(f"Description: {description}")
        print("="*60)
        
    elif args.input_file:
        # Batch prediction from file
        if not os.path.exists(args.input_file):
            print(f"Error: Input file '{args.input_file}' not found!")
            return
        
        print(f"Loading texts from {args.input_file}...")
        df = pd.read_csv(args.input_file)
        
        if 'text' not in df.columns:
            print("Error: Input file must contain a 'text' column!")
            return
        
        texts = df['text'].tolist()
        print(f"Found {len(texts)} texts to predict")
        
        # Make predictions
        predictions = predict_batch(texts, model, tokenizer, config, device)
        
        # Add predictions to dataframe
        df['complexity_score'] = predictions
        
        # Add interpretation
        levels = []
        descriptions = []
        for pred in predictions:
            level, desc = interpret_complexity(pred)
            levels.append(level)
            descriptions.append(desc)
        
        df['complexity_level'] = levels
        df['description'] = descriptions
        
        # Save results
        if args.output_file:
            df.to_csv(args.output_file, index=False)
            print(f"Results saved to {args.output_file}")
        else:
            # Print results
            print("\n" + "="*80)
            print("BATCH PREDICTION RESULTS")
            print("="*80)
            for i, (text, score, level) in enumerate(zip(texts, predictions, levels)):
                print(f"\n{i+1}. Text: {text[:100]}{'...' if len(text) > 100 else ''}")
                print(f"   Score: {score:.4f} | Level: {level}")
            print("="*80)
            
            # Print summary statistics
            print(f"\nSummary Statistics:")
            print(f"Mean complexity: {np.mean(predictions):.4f}")
            print(f"Std complexity: {np.std(predictions):.4f}")
            print(f"Min complexity: {np.min(predictions):.4f}")
            print(f"Max complexity: {np.max(predictions):.4f}")
            
            # Level distribution
            level_counts = df['complexity_level'].value_counts()
            print(f"\nLevel Distribution:")
            for level, count in level_counts.items():
                print(f"  {level}: {count} texts")
    
    else:
        # Interactive mode
        print("\nInteractive Text Complexity Predictor")
        print("Enter text to predict complexity (or 'quit' to exit)")
        print("-" * 50)
        
        while True:
            text = input("\nEnter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not text:
                print("Please enter some text!")
                continue
            
            try:
                prediction = predict_complexity(text, model, tokenizer, config, device)
                level, description = interpret_complexity(prediction)
                
                print(f"\nComplexity Score: {prediction:.4f}")
                print(f"Level: {level}")
                print(f"Description: {description}")
                
                if args.verbose:
                    print(f"\nDetailed Analysis:")
                    print(f"  - Score range: 0.0 (simplest) to 1.0 (most complex)")
                    print(f"  - Your text falls in the {level} category")
                    print(f"  - This suggests the text is {description.lower()}")
                
            except Exception as e:
                print(f"Error predicting complexity: {e}")

if __name__ == "__main__":
    main() 