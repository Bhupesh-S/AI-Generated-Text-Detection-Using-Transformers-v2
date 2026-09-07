# AI-Generated Text Detection Using Transformers

A research pipeline for classifying text as human-written or AI-generated. The project compares traditional machine-learning baselines with Transformer models and combines their predictions in a hybrid ensemble.

## What This Project Does

- Cleans and consolidates human and AI-generated text datasets.
- Trains TF-IDF and handcrafted-feature baselines.
- Fine-tunes RoBERTa and DeBERTa sequence classifiers.
- Combines base-model probabilities with an ensemble meta-classifier.
- Produces evaluation reports, figures, metrics, and diagnostic outputs.

## Architecture

1. **Preprocessing**: normalizes text, removes null bytes, handles whitespace, and filters very short samples.
2. **Feature engineering**: creates TF-IDF vectors and statistical or linguistic features.
3. **Baseline models**: Logistic Regression, Multinomial Naive Bayes, Linear SVM, Random Forest, and XGBoost.
4. **Transformer models**: RoBERTa-base and DeBERTa-v3-base fine-tuned for binary classification.
5. **Ensemble**: trains a meta-classifier on predictions from the base models.

## Repository Layout

```text
.
├── data_loader/              Dataset loading utilities
├── feature_engineering/      Handcrafted and TF-IDF features
├── models/                   Baseline, Transformer, and ensemble code
├── preprocessing/            Dataset cleaning and preparation
├── reports/                  EDA and evaluation reports
├── outputs/                  Generated metrics, figures, and model artifacts
├── requirements.txt          Python dependencies
├── clean_dataset.py         Dataset cleaning entry point
├── create_v2_1_splits.py    Dataset split creation
├── train_roberta.py         RoBERTa training
├── train_deberta.py         DeBERTa training
├── app.py                   Application entry point
└── Project.md               Detailed project documentation
```

## Requirements

- Python 3.12
- PyTorch
- Hugging Face Transformers
- scikit-learn
- XGBoost
- pandas and NumPy
- A CUDA-capable GPU is recommended for Transformer training.

Install the dependencies in a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data

The expected processed dataset is:

```text
datasets/merged/final_dataset.csv
```

The main columns are:

- `text`: document content.
- `label`: binary class, where `0` represents human text and `1` represents AI-generated text.
- `source`: originating dataset or generator.

Large datasets and generated model artifacts may be excluded from version control. Place them in the paths expected by the individual scripts before running the pipeline.

## Typical Workflow

Run the stages in this order:

```powershell
python preprocessing/preprocess.py
python models/baseline/train.py
python train_roberta.py
python train_deberta.py
python models/ensemble/generate_base_predictions.py
python models/ensemble/train.py
```

Additional scripts support dataset balancing, split generation, evaluation, auditing, and diagnostics:

```powershell
python create_v2_1_splits.py
python balance_dataset.py
python evaluate_v2_1_final.py
python dataset_audit.py
```

Check each script's configuration and input paths before running it. Training scripts can require substantial disk space, system memory, and GPU memory.

## Data Splits and Training Defaults

- Transformer split: 70% train, 15% validation, 15% test.
- Baseline and feature-engineering split: 70% train, 10% validation, 20% test.
- Transformer maximum sequence length: 512 tokens.
- Typical Transformer batch size: 8.
- Typical learning rate: `2e-5`.
- Typical random seed: `42`.
- TF-IDF uses unigrams and bigrams with up to 50,000 features.

The split strategies are currently different, so results from the baseline and Transformer pipelines should be compared with that limitation in mind.

## Known Limitations

- Transformer training is constrained by GPU memory and may require small batches.
- Loading complete CSV files with pandas can become memory-intensive as datasets grow.
- Some preprocessing logic is duplicated across scripts.
- Reproducibility depends on using the same dataset versions, splits, configuration, and model checkpoints.

## Further Documentation

See [Project.md](Project.md) for the full architecture, data pipeline, model details, hyperparameters, limitations, and future development notes.
