# Text Complexity Estimator

A machine learning project that fine-tunes a transformer model to predict text complexity scores as a regression task.

## 🧠 Concept
Fine-tune a transformer (e.g., bert-base-uncased) to predict text complexity scores (like readability, syntactic depth) as a regression task.

## 🎯 Goals
- **Input**: A sentence or paragraph
- **Output**: A score (e.g., Flesch-Kincaid, CEFR levels, or 0–1 complexity)
- Fine-tune a model to learn these scores from text

## 🛠️ Tech Stack
| Component | Tool |
|-----------|------|
| Model | bert-base-uncased or DistilBERT |
| Framework | Hugging Face Transformers |
| Dataset | CommonLit Readability |
| Metric | MSE, R² |

## 📂 Project Structure
```
text-complexity-predictor/
│
├── data/
│   └── readability.csv
│
├── model/
│   └── best/
│
├── train.py
├── evaluate.py
├── predict.py
├── app.py
├── README.md
└── requirements.txt
```

## 🚀 Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download the CommonLit Readability dataset and place it in `data/readability.csv`

3. Train the model:
```bash
python train.py
```

4. Evaluate the model:
```bash
python evaluate.py
```

5. Make predictions:
```bash
python predict.py --text "Your text here"
```

6. Run the web app:
```bash
python app.py
```

## 📊 Model Performance
The model is evaluated using:
- Mean Squared Error (MSE)
- R-squared (R²) coefficient
- Root Mean Squared Error (RMSE)

## 🤝 Contributing
Feel free to contribute to this project by submitting issues or pull requests. 