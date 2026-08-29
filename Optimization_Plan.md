# Optimization Plan: AI-Generated Text Detection

## 1. VRAM & Training Speed (High Impact)
- **Gradient Accumulation**: Implement in 	rain_roberta.py and 	rain_deberta.py to simulate larger batches (e.g., effective batch size 32) while keeping physical batch size low to fit in 4GB VRAM.
- **LoRA (Low-Rank Adaptation)**: Use peft library to fine-tune only a small subset of parameters. This will drastically reduce VRAM usage and training time while maintaining accuracy.
- **Unified Data Splitting**: Move splitting logic to a central utility to ensure all models are evaluated on the exact same test set.

## 2. Dataloader & I/O (Medium Impact)
- **Tuning Trainer Args**: Set
um_workers=8 and prefetch_factor=2 in TrainingArguments.
- **Pinned Memory**: Ensure pin_memory=True is used if using custom data loaders.
- **Dataset Caching**: Save tokenized datasets to disk to avoid re-tokenizing on every run.

## 3. Feature Engineering & Baselines (Low Impact)
- **Feature Pruning**: Evaluate if max_features=50,000 is necessary or if 10,000-20,000 suffices.
- **XGBoost Tuning**: Optimize
_jobs and 	ree_method for the RTX 3050.

## 4. Code Quality & Maintenance
- **Deduplicate Preprocessing**: Merge preprocessing/preprocess.py and eature_engineering/preprocessing.py into a single module.
- **Unified Config**: Replace local CONFIG dictionaries with a central config.yaml or config.py.
