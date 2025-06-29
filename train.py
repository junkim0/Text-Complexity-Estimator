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
    """BERT-based model for text complexity prediction"""
    
    def __init__(self, model_name='bert-base-uncased', dropout=0.1):
        super(TextComplexityModel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
    
    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.regressor(pooled_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.MSELoss()
            loss = loss_fct(logits.squeeze(), labels)
        
        return loss, logits.squeeze()

def load_data(data_path):
    """Load and preprocess the CommonLit dataset"""
    print("Loading dataset...")
    
    # For demo purposes, create a synthetic dataset
    # In practice, you would load the actual CommonLit dataset
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic texts with varying complexity
    simple_texts = [
        "The cat sat on the mat.",
        "I like to read books.",
        "The sun is bright today.",
        "She walks to school every day.",
        "The dog runs in the park."
    ]
    
    complex_texts = [
        "The intricate mechanisms underlying quantum entanglement continue to perplex even the most distinguished physicists in the field.",
        "Socioeconomic disparities manifest themselves through multifaceted channels, perpetuating cycles of inequality across generations.",
        "The epistemological foundations of modern science rest upon empirical observation and rigorous methodological frameworks.",
        "Constitutional jurisprudence necessitates careful consideration of both textual interpretation and historical context.",
        "Metacognitive strategies enable learners to monitor and regulate their cognitive processes effectively."
    ]
    
    texts = []
    targets = []
    
    for _ in range(n_samples // 2):
        # Simple texts (lower complexity scores)
        text = np.random.choice(simple_texts)
        texts.append(text)
        targets.append(np.random.uniform(0.1, 0.4))
        
        # Complex texts (higher complexity scores)
        text = np.random.choice(complex_texts)
        texts.append(text)
        targets.append(np.random.uniform(0.6, 0.9))
    
    # Add some medium complexity texts
    medium_texts = [
        "The weather forecast predicts rain for tomorrow.",
        "Students should complete their homework assignments.",
        "The restaurant serves delicious Italian food.",
        "Technology continues to advance rapidly.",
        "Exercise is important for maintaining good health."
    ]
    
    for _ in range(n_samples // 4):
        text = np.random.choice(medium_texts)
        texts.append(text)
        targets.append(np.random.uniform(0.4, 0.6))
    
    return pd.DataFrame({
        'text': texts,
        'target': targets
    })

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
    if os.path.exists(args.data_path):
        df = pd.read_csv(args.data_path)
    else:
        print("Dataset not found, creating synthetic dataset...")
        df = load_data(args.data_path)
        df.to_csv(args.data_path, index=False)
    
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