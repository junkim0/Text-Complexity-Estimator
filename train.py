import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class TextComplexityDataset(Dataset):
    """Dataset class for text complexity prediction"""
    
    def __init__(self, texts, targets, tokenizer, max_length=512):
        self.texts = texts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        target = float(self.targets[idx])
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(target, dtype=torch.float)
        }

class TextComplexityModel(nn.Module):
    """Enhanced BERT-based model for text complexity prediction"""
    
    def __init__(self, model_name='bert-base-uncased', dropout=0.2):
        super(TextComplexityModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Enhanced architecture with more layers and attention
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 768),
            nn.LayerNorm(768),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multiple regression heads for better complexity prediction
        self.complexity_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # Additional features for archaic language detection
        self.archaic_detector = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        # Combine features
        self.final_regressor = nn.Sequential(
            nn.Linear(2, 32),  # 1 from complexity + 1 from archaic
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        # Extract features
        features = self.feature_extractor(pooled_output)
        
        # Get complexity prediction
        complexity_pred = self.complexity_head(features)
        
        # Get archaic language detection
        archaic_score = self.archaic_detector(features)
        
        # Combine predictions
        combined_features = torch.cat([complexity_pred, archaic_score], dim=1)
        final_prediction = self.final_regressor(combined_features)
        
        # Boost score if archaic language is detected
        final_prediction = final_prediction + (archaic_score * 0.1)
        
        # Ensure prediction is in [0, 1] range
        final_prediction = torch.sigmoid(final_prediction)
        
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(final_prediction.squeeze(), labels)
        
        return loss, final_prediction.squeeze()

def load_data(data_path):
    """Load and preprocess the enhanced dataset"""
    print("Loading enhanced dataset...")
    
    try:
        # Try to load the enhanced dataset first
        df = pd.read_csv('data/enhanced_readability.csv')
        print(f"Loaded enhanced dataset with {len(df)} samples")
    except FileNotFoundError:
        # Fallback to original dataset
        print("Enhanced dataset not found, using original dataset...")
        df = pd.read_csv(data_path)
        print(f"Loaded original dataset with {len(df)} samples")
    
    return df

def train_model(model, train_dataloader, val_dataloader, device, args):
    """Train the model"""
    print("Starting training...")
    
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=total_steps
    )
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Training
        model.train()
        total_train_loss = 0
        train_progress_bar = tqdm(train_dataloader, desc="Training")
        
        for batch in train_progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            model.zero_grad()
            loss, _ = model(input_ids, attention_mask, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            train_progress_bar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        total_val_loss = 0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            val_progress_bar = tqdm(val_dataloader, desc="Validation")
            for batch in val_progress_bar:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                loss, logits = model(input_ids, attention_mask, labels)
                total_val_loss += loss.item()
                
                val_predictions.extend(logits.cpu().numpy())
                val_targets.extend(labels.cpu().numpy())
                
                val_progress_bar.set_postfix({'loss': loss.item()})
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)
        
        # Calculate metrics
        val_mse = mean_squared_error(val_targets, val_predictions)
        val_r2 = r2_score(val_targets, val_predictions)
        
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        print(f"Val MSE: {val_mse:.4f}")
        print(f"Val R²: {val_r2:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(args.model_dir, 'best_model.pth'))
            print("New best model saved!")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.scatter(val_targets, val_predictions, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--', lw=2)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.title('Prediction vs True Values')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.model_dir, 'training_curves.png'))
    plt.close()
    
    return model

def main():
    parser = argparse.ArgumentParser(description='Train Text Complexity Model')
    parser.add_argument('--data_path', type=str, default='data/readability.csv',
                       help='Path to the dataset')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                       help='Pretrained model name')
    parser.add_argument('--model_dir', type=str, default='model/best',
                       help='Directory to save the model')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='Learning rate')
    parser.add_argument('--max_length', type=int, default=512,
                       help='Maximum sequence length')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Test set size')
    parser.add_argument('--val_size', type=float, default=0.1,
                       help='Validation set size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create model directory
    os.makedirs(args.model_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    df = load_data(args.data_path)
    
    # Split data
    train_df, temp_df = train_test_split(df, test_size=args.test_size + args.val_size, random_state=args.seed)
    val_df, test_df = train_test_split(temp_df, test_size=args.test_size/(args.test_size + args.val_size), random_state=args.seed)
    
    print(f"Train set: {len(train_df)} samples")
    print(f"Validation set: {len(val_df)} samples")
    print(f"Test set: {len(test_df)} samples")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Create datasets
    train_dataset = TextComplexityDataset(
        train_df['text'].values, 
        train_df['target'].values, 
        tokenizer, 
        args.max_length
    )
    
    val_dataset = TextComplexityDataset(
        val_df['text'].values, 
        val_df['target'].values, 
        tokenizer, 
        args.max_length
    )
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Initialize model
    model = TextComplexityModel(args.model_name)
    model.to(device)
    
    # Train model
    trained_model = train_model(model, train_dataloader, val_dataloader, device, args)
    
    # Save tokenizer
    tokenizer.save_pretrained(args.model_dir)
    
    # Save training configuration
    config = {
        'model_name': args.model_name,
        'max_length': args.max_length,
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs
    }
    
    with open(os.path.join(args.model_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nTraining completed! Model saved to {args.model_dir}")
    print("You can now use the model for predictions with predict.py")

if __name__ == "__main__":
    main() 