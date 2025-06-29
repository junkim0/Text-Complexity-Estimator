import os
import pandas as pd
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(__file__), 'readability.csv')

# Kaggle import is optional and may not be available
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    kaggle_available = True
except ImportError:
    kaggle_available = False

def download_commonlit():
    if not kaggle_available:
        print("Kaggle API not available. Skipping download.")
        return False
    try:
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files('c/commonlitreadabilityprize', path='.', unzip=True)
        # Read the full CSV into a DataFrame
        df_full = pd.read_csv('train.csv').copy()
        # Only keep necessary columns
        if 'excerpt' in df_full.columns and 'target' in df_full.columns:
            df = pd.DataFrame({'text': df_full['excerpt'], 'target': df_full['target']})
            df.to_csv(DATA_PATH, index=False)
            print(f"Downloaded and saved dataset to {DATA_PATH}")
            return True
        else:
            print("train.csv does not have the required columns.")
            return False
    except Exception as e:
        print(f"Failed to download dataset: {e}")
        return False

def generate_synthetic():
    print("Generating synthetic dataset...")
    np.random.seed(42)
    n_samples = 1000
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
        texts.append(np.random.choice(simple_texts))
        targets.append(np.random.uniform(0.1, 0.4))
        texts.append(np.random.choice(complex_texts))
        targets.append(np.random.uniform(0.6, 0.9))
    medium_texts = [
        "The weather forecast predicts rain for tomorrow.",
        "Students should complete their homework assignments.",
        "The restaurant serves delicious Italian food.",
        "Technology continues to advance rapidly.",
        "Exercise is important for maintaining good health."
    ]
    for _ in range(n_samples // 4):
        texts.append(np.random.choice(medium_texts))
        targets.append(np.random.uniform(0.4, 0.6))
    df = pd.DataFrame({'text': texts, 'target': targets})
    df.to_csv(DATA_PATH, index=False)
    print(f"Synthetic dataset saved to {DATA_PATH}")

def main():
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    if os.path.exists(DATA_PATH):
        print(f"Dataset already exists at {DATA_PATH}")
        return
    if not download_commonlit():
        generate_synthetic()

if __name__ == "__main__":
    main() 