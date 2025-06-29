# Vercel Deployment Guide

This guide will help you deploy the Text Complexity Predictor to Vercel.

## 🚀 Quick Deploy

### Option 1: Deploy from GitHub (Recommended)

1. **Fork/Clone the Repository**
   ```bash
   git clone https://github.com/junkim0/Text-Complexity-Estimator.git
   cd Text-Complexity-Estimator
   ```

2. **Create Vercel Account**
   - Go to [vercel.com](https://vercel.com)
   - Sign up with GitHub account

3. **Deploy to Vercel**
   - Click "New Project" in Vercel dashboard
   - Import your GitHub repository
   - Vercel will automatically detect it's a Python project
   - Click "Deploy"

### Option 2: Manual Deploy

1. **Install Vercel CLI**
   ```bash
   npm install -g vercel
   ```

2. **Login to Vercel**
   ```bash
   vercel login
   ```

3. **Deploy**
   ```bash
   vercel
   ```

## 📁 Project Structure for Vercel

```
text-complexity-predictor/
├── app.py                 # Main Flask application
├── vercel.json           # Vercel configuration
├── requirements.txt      # Python dependencies
├── model/
│   └── best/            # Trained model files
├── templates/
│   └── index.html       # Web interface
└── data/                # Dataset files
```

## ⚙️ Configuration Files

### vercel.json
```json
{
  "version": 2,
  "builds": [
    {
      "src": "app.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "app.py"
    }
  ],
  "env": {
    "PYTHONPATH": "."
  },
  "functions": {
    "app.py": {
      "maxDuration": 30
    }
  }
}
```

### requirements.txt
```
torch==2.0.1
transformers==4.30.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
flask==2.3.3
werkzeug==2.3.7
tqdm==4.65.0
requests==2.31.0
```

## 🔧 Environment Variables

No environment variables are required for basic deployment. The model files are included in the repository.

## 📊 Model Files

The trained model files are included in the repository:
- `model/best/best_model.pth` - Trained model weights
- `model/best/config.json` - Model configuration
- `model/best/tokenizer.json` - Tokenizer files

## 🌐 Deployment Features

### Enhanced Web Interface
- **Detailed Score System**: Comprehensive explanation of complexity levels
- **Shakespeare Detection**: Specialized analysis for archaic language
- **Multiple Input Methods**: Single text, batch analysis, file upload
- **Responsive Design**: Works on desktop and mobile devices

### API Endpoints
- `GET /` - Main web interface
- `POST /predict` - Single text prediction
- `POST /batch_predict` - Batch text prediction
- `POST /upload` - CSV file upload
- `GET /health` - Health check

## 🚀 Post-Deployment

### 1. Verify Deployment
- Check the health endpoint: `https://your-app.vercel.app/health`
- Test the web interface: `https://your-app.vercel.app/`

### 2. Test Functionality
```bash
# Test single prediction
curl -X POST https://your-app.vercel.app/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "To be, or not to be, that is the question"}'

# Test health check
curl https://your-app.vercel.app/health
```

### 3. Custom Domain (Optional)
- Go to Vercel dashboard
- Select your project
- Go to "Settings" → "Domains"
- Add your custom domain

## 🔍 Troubleshooting

### Common Issues

1. **Model Loading Error**
   - Ensure model files are in `model/best/` directory
   - Check file permissions
   - Verify model files are committed to repository

2. **Memory Issues**
   - Vercel has memory limits
   - Model is optimized for CPU inference
   - Consider using smaller model variants if needed

3. **Timeout Issues**
   - Increase `maxDuration` in vercel.json
   - Optimize model loading
   - Use caching strategies

### Performance Optimization

1. **Model Loading**
   - Model loads once on cold start
   - Subsequent requests are faster
   - Consider model caching strategies

2. **Response Time**
   - Typical response time: 1-3 seconds
   - Depends on text length and complexity
   - Batch processing may take longer

## 📈 Monitoring

### Vercel Analytics
- View deployment logs in Vercel dashboard
- Monitor function execution times
- Track error rates and performance

### Health Monitoring
```bash
# Regular health checks
curl https://your-app.vercel.app/health
```

## 🔄 Updates

### Updating the Model
1. Train new model locally
2. Replace `model/best/` files
3. Commit and push to GitHub
4. Vercel will automatically redeploy

### Updating Dependencies
1. Update `requirements.txt`
2. Commit and push changes
3. Vercel will rebuild with new dependencies

## 🎯 Best Practices

1. **Keep Model Files Small**
   - Use model compression techniques
   - Consider quantization for smaller models
   - Optimize for inference speed

2. **Error Handling**
   - Implement proper error responses
   - Log errors for debugging
   - Provide user-friendly error messages

3. **Security**
   - Validate input data
   - Implement rate limiting if needed
   - Use HTTPS (automatic with Vercel)

## 📞 Support

If you encounter issues:
1. Check Vercel deployment logs
2. Verify model files are present
3. Test locally first
4. Check the health endpoint

## 🎉 Success!

Once deployed, your Text Complexity Predictor will be available at:
`https://your-app.vercel.app/`

The enhanced web interface includes:
- ✅ Detailed complexity score explanations
- ✅ Shakespeare and archaic language detection
- ✅ Multiple input methods
- ✅ Responsive design
- ✅ Professional UI/UX 