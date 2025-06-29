import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from train import TextComplexityModel, TextComplexityDataset, load_data

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

def evaluate_model(model, test_dataloader, device):
    """Evaluate the model on test set"""
    print("Evaluating model on test set...")
    
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            _, logits = model(input_ids, attention_mask, labels)
            
            predictions.extend(logits.cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    return np.array(predictions), np.array(targets)

def calculate_metrics(predictions, targets):
    """Calculate evaluation metrics"""
    mse = mean_squared_error(targets, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(targets, predictions)
    r2 = r2_score(targets, predictions)
    
    # Calculate additional metrics
    mape = np.mean(np.abs((targets - predictions) / targets)) * 100
    
    metrics = {
        'MSE': float(mse),
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R²': float(r2),
        'MAPE': float(mape)
    }
    
    return metrics

def plot_results(predictions, targets, save_dir):
    """Create evaluation plots"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Prediction vs True Values
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(targets, predictions, alpha=0.6, s=20)
    plt.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Prediction vs True Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Residuals Plot
    plt.subplot(2, 2, 2)
    residuals = predictions - targets
    plt.scatter(predictions, residuals, alpha=0.6, s=20)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predictions')
    plt.ylabel('Residuals')
    plt.title('Residuals Plot')
    plt.grid(True, alpha=0.3)
    
    # 3. Distribution of Predictions vs Targets
    plt.subplot(2, 2, 3)
    plt.hist(targets, bins=30, alpha=0.7, label='True Values', density=True)
    plt.hist(predictions, bins=30, alpha=0.7, label='Predictions', density=True)
    plt.xlabel('Complexity Score')
    plt.ylabel('Density')
    plt.title('Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 4. Error Distribution
    plt.subplot(2, 2, 4)
    plt.hist(residuals, bins=30, alpha=0.7, color='orange')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Error Distribution')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'evaluation_results.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Additional detailed plots
    # 5. Error vs True Values
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(targets, np.abs(residuals), alpha=0.6, s=20)
    plt.xlabel('True Values')
    plt.ylabel('Absolute Error')
    plt.title('Absolute Error vs True Values')
    plt.grid(True, alpha=0.3)
    
    # 6. Error vs Predictions
    plt.subplot(1, 2, 2)
    plt.scatter(predictions, np.abs(residuals), alpha=0.6, s=20)
    plt.xlabel('Predictions')
    plt.ylabel('Absolute Error')
    plt.title('Absolute Error vs Predictions')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'error_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_predictions(predictions, targets, save_dir):
    """Analyze prediction patterns"""
    residuals = predictions - targets
    
    analysis = {
        'total_samples': int(len(predictions)),
        'mean_prediction': float(np.mean(predictions)),
        'mean_target': float(np.mean(targets)),
        'std_prediction': float(np.std(predictions)),
        'std_target': float(np.std(targets)),
        'correlation': float(np.corrcoef(predictions, targets)[0, 1]),
        'max_error': float(np.max(np.abs(residuals))),
        'min_error': float(np.min(np.abs(residuals))),
        'error_std': float(np.std(residuals))
    }
    
    # Save analysis
    with open(os.path.join(save_dir, 'analysis.json'), 'w') as f:
        json.dump(analysis, f, indent=2)
    
    return analysis

def main():
    parser = argparse.ArgumentParser(description='Evaluate Text Complexity Model')
    parser.add_argument('--model_dir', type=str, default='model/best',
                       help='Directory containing the trained model')
    parser.add_argument('--data_path', type=str, default='data/readability.csv',
                       help='Path to the dataset')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Validation set size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load trained model
    print("Loading trained model...")
    model, tokenizer, config = load_trained_model(args.model_dir, device)
    
    # Load data
    if os.path.exists(args.data_path):
        df = pd.read_csv(args.data_path)
    else:
        print("Dataset not found, creating synthetic dataset...")
        df = load_data(args.data_path)
        df.to_csv(args.data_path, index=False)
    
    # Split data (same as training)
    train_df, temp_df = train_test_split(df, test_size=args.test_size + args.val_size, random_state=args.seed)
    val_df, test_df = train_test_split(temp_df, test_size=args.test_size/(args.test_size + args.val_size), random_state=args.seed)
    
    print(f"Test set: {len(test_df)} samples")
    
    # Create test dataset
    test_dataset = TextComplexityDataset(
        test_df['text'].values, 
        test_df['target'].values, 
        tokenizer, 
        config['max_length']
    )
    
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Evaluate model
    predictions, targets = evaluate_model(model, test_dataloader, device)
    
    # Calculate metrics
    metrics = calculate_metrics(predictions, targets)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print("="*50)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save metrics
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions
    results_df = pd.DataFrame({
        'true_values': targets,
        'predictions': predictions,
        'errors': predictions - targets,
        'absolute_errors': np.abs(predictions - targets)
    })
    results_df.to_csv(os.path.join(args.output_dir, 'predictions.csv'), index=False)
    
    # Create plots
    print("Creating evaluation plots...")
    plot_results(predictions, targets, args.output_dir)
    
    # Analyze predictions
    print("Analyzing predictions...")
    analysis = analyze_predictions(predictions, targets, args.output_dir)
    
    # Print analysis summary
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    print(f"Total samples: {analysis['total_samples']}")
    print(f"Mean prediction: {analysis['mean_prediction']:.4f}")
    print(f"Mean target: {analysis['mean_target']:.4f}")
    print(f"Correlation: {analysis['correlation']:.4f}")
    print(f"Max absolute error: {analysis['max_error']:.4f}")
    print(f"Error standard deviation: {analysis['error_std']:.4f}")
    print("="*50)
    
    print(f"\nEvaluation completed! Results saved to {args.output_dir}")
    print("Files created:")
    print(f"- {args.output_dir}/metrics.json")
    print(f"- {args.output_dir}/predictions.csv")
    print(f"- {args.output_dir}/evaluation_results.png")
    print(f"- {args.output_dir}/error_analysis.png")
    print(f"- {args.output_dir}/analysis.json")

if __name__ == "__main__":
    main() 