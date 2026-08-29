import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from data_utils import get_unified_splits
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix, roc_curve, auc
from scipy.special import softmax
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed
)

# Configuration
CONFIG = {
    "model_name": "microsoft/deberta-v3-base",
    "data_path": "datasets/merged/final_dataset.csv",
    "output_dir": "models/deberta",
    "reports_dir": "reports/deberta",
    "max_length": 512,
    "batch_size": 4,           # Micro-batch size 4 fits easily on RTX 3050
    "learning_rate": 1.5e-5,   # DeBERTa-v3 converges fast; 1.5e-5 is more stable
    "epochs": 5,           
    "patience": 2,         
    "seed": 42,
    "test_size": 0.10,     # Standardized 80/10/10 split
    "val_size": 0.10       # Standardized 80/10/10 split
}

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

def setup_directories():
    """Ensure output directories exist."""
    Path(CONFIG["output_dir"]).mkdir(parents=True, exist_ok=True)
    Path(CONFIG["reports_dir"]).mkdir(parents=True, exist_ok=True)

def load_and_split_data(file_path: str) -> DatasetDict:
    """Load dataset from CSV, check validity, and split into Train/Val/Test."""
    logger.info(f"Loading dataset from {file_path}")
    
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}")
        
    # Git LFS Pointer check
    if path.stat().st_size < 10000:
        with open(path, 'r') as f:
            content = f.read(100)
            if "git-lfs" in content:
                raise ValueError(f"The file {file_path} is a Git LFS pointer. Please run `git lfs pull` first to download the actual data.")

    df = pd.read_csv(file_path)
    if 'text' not in df.columns or 'label' not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")
        
    df = df.dropna(subset=['text', 'label'])
    
    # Deduplicate raw dataset to prevent leakage
    before_dedup = len(df)
    df = df.drop_duplicates(subset=['text']).reset_index(drop=True)
    deduped = before_dedup - len(df)
    if deduped > 0:
        logger.info(f"Removed {deduped:,} duplicate rows during loading.")

    df['text'] = df['text'].astype(str)
    df['label'] = df['label'].astype(int)
    
    # Unified splitting from data_utils
    train_df, val_df, test_df = get_unified_splits(df, test_size=CONFIG["test_size"], val_size=CONFIG["val_size"], seed=CONFIG["seed"])
    
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "validation": Dataset.from_pandas(val_df),
        "test": Dataset.from_pandas(test_df)
    })
    return dataset

def compute_metrics(eval_pred):
    """Calculate Accuracy, Precision, Recall, F1, and ROC-AUC."""
    logits, labels = eval_pred
    
    # Apply softmax to get probabilities
    probs = softmax(logits, axis=-1)
    
    # Get predicted classes
    preds = np.argmax(logits, axis=-1)
    
    # Metrics
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='binary', zero_division=0
    )
    
    # ROC-AUC using probabilities of the positive class (class 1: AI-generated)
    roc_auc = roc_auc_score(labels, probs[:, 1])
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc
    }

def plot_confusion_matrix(labels, preds, output_path: str):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Human (0)', 'AI (1)'], yticklabels=['Human (0)', 'AI (1)'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (DeBERTa-v3)')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Confusion matrix saved to {output_path}")

def plot_learning_curves(log_history, output_path: str):
    """Extract metrics from Trainer log history and plot Loss and Accuracy."""
    train_loss = []
    eval_loss = []
    eval_acc = []
    epochs = []
    
    for entry in log_history:
        if 'loss' in entry and 'epoch' in entry:
            train_loss.append((entry['epoch'], entry['loss']))
        elif 'eval_loss' in entry and 'epoch' in entry:
            eval_loss.append((entry['epoch'], entry['eval_loss']))
            eval_acc.append((entry['epoch'], entry['eval_accuracy']))

    if not train_loss or not eval_loss:
        logger.warning("Not enough log history to plot learning curves.")
        return

    # Convert to DataFrames for easier plotting
    df_train = pd.DataFrame(train_loss, columns=['epoch', 'loss'])
    df_eval = pd.DataFrame(eval_loss, columns=['epoch', 'loss'])
    df_acc = pd.DataFrame(eval_acc, columns=['epoch', 'accuracy'])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot Loss
    ax1.plot(df_train['epoch'], df_train['loss'], label='Train Loss')
    ax1.plot(df_eval['epoch'], df_eval['loss'], label='Validation Loss', marker='o')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    
    # Plot Accuracy
    ax2.plot(df_acc['epoch'], df_acc['accuracy'], label='Validation Accuracy', color='green', marker='o')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Learning curves saved to {output_path}")

def plot_roc_curve(labels, probs, output_path: str):
    """Plot and save ROC Curve."""
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - DeBERTa-v3')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"ROC curve saved to {output_path}")

def generate_markdown_report(metrics: dict, config: dict, output_path: str):
    """Generate a clean markdown report of the experiment."""
    report = f"""# DeBERTa-v3 Fine-Tuning Report for AI-Generated Text Detection

## Configuration
- **Model**: `{config['model_name']}`
- **Dataset**: `{config['data_path']}`
- **Max Sequence Length**: `{config['max_length']}`
- **Batch Size**: `{config['batch_size']}`
- **Learning Rate**: `{config['learning_rate']}`
- **Max Epochs**: `{config['epochs']}`
- **Early Stopping Patience**: `{config['patience']}`

## Test Set Performance Metrics
| Metric | Score |
|---|---|
| **Accuracy** | {metrics.get('test_accuracy', 0):.4f} |
| **Precision** | {metrics.get('test_precision', 0):.4f} |
| **Recall** | {metrics.get('test_recall', 0):.4f} |
| **F1 Score** | {metrics.get('test_f1', 0):.4f} |
| **ROC-AUC** | {metrics.get('test_roc_auc', 0):.4f} |

## Visualizations
### Confusion Matrix
![Confusion Matrix](./confusion_matrix.png)

### ROC Curve
![ROC Curve](./roc_curve.png)

### Learning Curves
![Learning Curves](./learning_curves.png)
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report generated at {output_path}")

def main():
    set_seed(CONFIG["seed"])
    setup_directories()
    
    # Device setup (Auto GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # 1. Load Data
    try:
        datasets = load_and_split_data(CONFIG["data_path"])
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    # 2. Load Tokenizer
    logger.info(f"Loading tokenizer: {CONFIG['model_name']}")
    # Note: deberta-v3 requires the sentencepiece library installed
    try:
        tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    except ValueError as e:
        logger.error(f"Tokenizer load failed. You might need to install sentencepiece: pip install sentencepiece")
        raise e
    
    def tokenize_function(examples):
        # Dynamic padding: do not pad here, let the data collator do it batch-wise
        return tokenizer(
            examples['text'], 
            truncation=True, 
            padding=False, 
            max_length=CONFIG["max_length"]
        )
        
    logger.info("Tokenizing datasets...")
    tokenized_datasets = datasets.map(
        tokenize_function, 
        batched=True, 
        num_proc=4 if os.cpu_count() and os.cpu_count() >= 4 else 1
    )
    
    # 3. Load Model
    logger.info(f"Loading model: {CONFIG['model_name']}")
    model = AutoModelForSequenceClassification.from_pretrained(
        CONFIG["model_name"], 
        num_labels=2
    ).to(device)
    
    # 4. Define Training Arguments
    # DeBERTa-v3 relative positional embedding gradients are unstable in float16.
    # Use native bfloat16 on Ampere architectures to avoid NaNs.
    is_cuda = device.type == "cuda"
    use_bf16 = is_cuda
    fused_optim = "adamw_torch_fused" if is_cuda else "adamw_torch"
    
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=CONFIG["learning_rate"],
        per_device_train_batch_size=CONFIG["batch_size"],
        gradient_accumulation_steps=4,       # Effective batch size = 4 * 4 = 16
        per_device_eval_batch_size=CONFIG["batch_size"],
        num_train_epochs=CONFIG["epochs"],
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_strategy="steps",
        logging_steps=50,
        bf16=use_bf16,                       # BF16 for DeBERTa stability on RTX 3050
        fp16=False,                          # Explicitly disable FP16
        gradient_checkpointing=True,         # Optimize VRAM to fit in 4GB VRAM
        optim=fused_optim,                   # Speed up optimizer updates
        label_smoothing_factor=0.1,          # Reduce overconfidence
        dataloader_pin_memory=True,          # DMA acceleration
        report_to="none" 
    )
    
    # DataCollatorWithPadding for dynamic padding with 8-byte tensor alignment
    from transformers import DataCollatorWithPadding
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8
    )
    
    # 5. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        processing_class=tokenizer,          # Transformers v5 compatibility
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=CONFIG["patience"])]
    )
    
    # 6. Train Model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the best model explicitly
    logger.info(f"Saving best model to {CONFIG['output_dir']}")
    trainer.save_model(CONFIG["output_dir"])
    
    # 7. Evaluate on Test Set
    logger.info("Evaluating on Test Set...")
    test_results = trainer.evaluate(eval_dataset=tokenized_datasets["test"], metric_key_prefix="test")
    logger.info(f"Test Results: {test_results}")
    
    # Predict to get Confusion Matrix and ROC Curve
    predictions = trainer.predict(tokenized_datasets["test"])
    logits = predictions.predictions
    labels = predictions.label_ids
    
    preds = np.argmax(logits, axis=-1)
    probs = softmax(logits, axis=-1)[:, 1] # Probabilities for the positive class (AI-generated)
    
    # 8. Generate Artifacts
    cm_path = f"{CONFIG['reports_dir']}/confusion_matrix.png"
    roc_path = f"{CONFIG['reports_dir']}/roc_curve.png"
    lc_path = f"{CONFIG['reports_dir']}/learning_curves.png"
    report_path = f"{CONFIG['reports_dir']}/deberta_report.md"
    
    plot_confusion_matrix(labels, preds, cm_path)
    plot_roc_curve(labels, probs, roc_path)
    plot_learning_curves(trainer.state.log_history, lc_path)
    generate_markdown_report(test_results, CONFIG, report_path)
    
    logger.info("DeBERTa-v3 Pipeline completed successfully!")

if __name__ == "__main__":
    main()

