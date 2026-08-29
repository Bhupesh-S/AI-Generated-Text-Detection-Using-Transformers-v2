"""
Pipeline smoke-test — run from repo root:
    python smoke_test.py
"""
import sys, os
sys.path.insert(0, '.')

errors = []

# 1. config
try:
    from feature_engineering.config import (
        DATASET_PATH, TFIDF_DIR, LABELS_DIR, TFIDF_PARAMS,
        TEST_SIZE, VALID_SIZE, RANDOM_STATE
    )
    assert DATASET_PATH.name == 'final_dataset.csv', f'wrong dataset: {DATASET_PATH}'
    assert TFIDF_PARAMS['max_features'] == 50_000
    assert TFIDF_PARAMS['ngram_range'] == (1, 2)
    assert TFIDF_PARAMS['min_df'] == 5
    assert TFIDF_PARAMS['max_df'] == 0.95
    print('[OK] config — paths and TF-IDF params correct')
except Exception as e:
    errors.append(f'config: {e}')

# 2. preprocessing
try:
    from feature_engineering.preprocessing import TextPreprocessor
    p = TextPreprocessor()
    out = p.clean_text('  Hello <b>World</b>! https://example.com   ')
    assert 'http' not in out
    assert '<b>' not in out
    assert out == out.lower()
    print(f'[OK] preprocessing — clean_text: "{out}"')
except Exception as e:
    errors.append(f'preprocessing: {e}')

# 3. utils
try:
    from feature_engineering.utils import extract_features
    feats = extract_features('The quick brown fox jumps. It is fast!')
    assert feats['word_count'] > 0
    assert 0 <= feats['lexical_diversity'] <= 1
    expected_keys = {
        'word_count','char_count','sentence_count','avg_word_length',
        'avg_sentence_length','vocab_size','lexical_diversity',
        'stopword_ratio','punctuation_ratio','digit_ratio','uppercase_ratio'
    }
    assert set(feats.keys()) == expected_keys, f"Feature mismatch: {set(feats.keys())}"
    print(f'[OK] utils — 11 features extracted correctly')
    for k, v in feats.items():
        print(f'       {k}: {v}')
except Exception as e:
    errors.append(f'utils: {e}')

# 4. vectorizer
try:
    from feature_engineering.vectorizer import TfidfVectorizerWrapper
    v = TfidfVectorizerWrapper(max_features=100, ngram_range=(1,2), min_df=1, max_df=1.0)
    corpus = ['hello world test', 'ai generated text detection', 'machine learning baseline']
    v.fit(corpus)
    mat = v.transform(corpus)
    assert mat.shape[0] == 3
    print(f'[OK] vectorizer — matrix shape: {mat.shape}')
except Exception as e:
    errors.append(f'vectorizer: {e}')

# 5. dataset path exists
try:
    from feature_engineering.config import DATASET_PATH
    assert DATASET_PATH.exists(), f'Dataset not found at {DATASET_PATH}'
    import pandas as pd
    df_peek = pd.read_csv(DATASET_PATH, usecols=['text','label'], nrows=5)
    assert list(df_peek.columns) == ['text', 'label']
    print(f'[OK] dataset — found at {DATASET_PATH}')
    print(f'       columns: {df_peek.columns.tolist()}')
    print(f'       label distribution (5 rows): {df_peek["label"].tolist()}')
except Exception as e:
    errors.append(f'dataset: {e}')

# 6. Full module imports
try:
    from feature_engineering.feature_engineering import FeatureEngineeringPipeline
    from models.baseline.evaluate  import ModelEvaluator
    from models.baseline.train     import BaselineModelTrainer
    from models.baseline.inference import ModelInference
    print('[OK] full imports — all modules loaded successfully')
except Exception as e:
    errors.append(f'full imports: {e}')

# 7. Directory structure
try:
    from pathlib import Path
    root = Path('.')
    required_dirs = [
        'Datasets/merged',
        'feature_engineering',
        'models/baseline',
        'outputs/tfidf',
        'outputs/models',
        'outputs/reports',
        'outputs/figures',
        'outputs/handcrafted',
        'outputs/labels',
        'logs',
    ]
    missing = [d for d in required_dirs if not (root / d).exists()]
    if missing:
        raise FileNotFoundError(f'Missing dirs: {missing}')
    print(f'[OK] directory structure — all {len(required_dirs)} directories present')
except Exception as e:
    errors.append(f'directory structure: {e}')

print()
if errors:
    print('=' * 60)
    print('FAILURES DETECTED:')
    for err in errors:
        print(f'  x  {err}')
    print('=' * 60)
    sys.exit(1)
else:
    print('=' * 60)
    print('All checks passed. Pipeline is ready to run.')
    print('=' * 60)
    print()
    print('Run the full pipeline with:')
    print('  python models/baseline/train.py')
    print()
    print('CLI options:')
    print('  --rerun-fe      Force re-run feature engineering even if artefacts exist')
    print('  --force-cpu     Disable GPU for XGBoost')
    print('  --n-jobs N      Number of parallel workers (-1 = all CPU cores)')
    print('  --xgboost-only  Train and evaluate only XGBoost (skip other models)')
