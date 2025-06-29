# Branch Structure and Deployment Strategy

This repository uses a dual-branch strategy to support both development and deployment needs.

## Branch Overview

### 🚀 `vercel-demo` - Lightweight Demo Version
- **Purpose**: Vercel deployment-ready version
- **Features**: Rule-based complexity estimation
- **Dependencies**: Minimal (pandas, numpy, flask, werkzeug, requests)
- **Size**: ~5MB (fits Vercel limits)
- **Deployment**: Auto-deploys to Vercel
- **URL**: [Live Demo](https://your-vercel-app.vercel.app)

### 🧠 `full-model` - Complete ML Version
- **Purpose**: Full-featured development version
- **Features**: BERT-based neural network complexity prediction
- **Dependencies**: Complete ML stack (PyTorch, Transformers, etc.)
- **Size**: ~500MB+ (includes trained model)
- **Deployment**: Local development, GitHub showcasing
- **Performance**: High accuracy with Shakespeare detection

### 📚 `main` - Stable Release
- **Purpose**: Main branch for stable releases
- **Content**: Currently mirrors the demo version
- **Usage**: Default branch for repository

## Development Workflow

### Working on Features
1. **Switch to full-model branch**:
   ```bash
   git checkout full-model
   ```

2. **Develop and test with full ML capabilities**:
   - Train models with `python train.py`
   - Evaluate with `python evaluate.py`
   - Test locally with `python app.py`

3. **Commit changes**:
   ```bash
   git add .
   git commit -m "Your feature description"
   git push origin full-model
   ```

### Updating Demo Version
When ready to update the deployed demo:

1. **Update rule-based logic in vercel-demo**:
   ```bash
   git checkout vercel-demo
   # Update api/index.py with improved rule-based estimation
   git commit -m "Update demo logic"
   git push origin vercel-demo
   ```

2. **Vercel auto-deploys** the updated demo

## File Differences

### `vercel-demo` Branch
```
api/index.py          # Lightweight rule-based estimator
requirements.txt      # 5 minimal dependencies
.vercelignore        # Excludes heavy files
vercel.json          # Deployment configuration
```

### `full-model` Branch
```
api/index.py          # Full BERT model integration
app.py                # Original Flask app
requirements.txt      # 12+ ML dependencies
model/                # Trained model files (422MB+)
data/                 # Training datasets
train.py              # Model training script
evaluate.py           # Model evaluation
```

## Deployment Status

- ✅ **Demo Version**: Deployed on Vercel (lightweight, fast)
- ✅ **Full Version**: Available on GitHub (complete functionality)
- ✅ **Both versions**: Maintain same API interface

## Performance Comparison

| Feature | Demo Version | Full Version |
|---------|-------------|--------------|
| **Deployment** | ✅ Vercel | ❌ Too large |
| **Speed** | Very Fast | Moderate |
| **Accuracy** | Good | Excellent |
| **Shakespeare Detection** | Pattern-based | ML-learned |
| **Dependencies** | 5 packages | 12+ packages |
| **Size** | ~5MB | ~500MB |

## Next Steps

1. Continue developing features in `full-model` branch
2. Test and refine ML model accuracy
3. Periodically update `vercel-demo` with improved rule-based logic
4. Consider model optimization techniques for potential full deployment 