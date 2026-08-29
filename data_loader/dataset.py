import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict, Any

class AIGeneratedTextDataset(Dataset):
    """
    Custom PyTorch Dataset for AI-generated text detection.
    Compatible with Hugging Face Transformers.
    """

    def __init__(self, dataframe: pd.DataFrame, tokenizer=None, max_length: int = 512):
        """
        Initialize the dataset.

        Args:
            dataframe (pd.DataFrame): DataFrame containing 'text' and 'label' columns.
            tokenizer: Optional Hugging Face tokenizer instance.
            max_length (int): Maximum sequence length for the tokenizer.
        """
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['label'].tolist()

        # Optional metadata fields if available
        self.source_datasets = dataframe['source_dataset'].tolist() if 'source_dataset' in dataframe.columns else None
        self.source_models = dataframe['source_model'].tolist() if 'source_model' in dataframe.columns else None
        self.filenames = dataframe['filename'].tolist() if 'filename' in dataframe.columns else None

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Fetch the text and label at the given index.
        If a tokenizer is provided, return the encoded inputs.

        Args:
            idx (int): Index of the sample to fetch.

        Returns:
            Dict[str, Any]: A dictionary containing the features (and optionally input_ids/attention_mask).
        """
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Create the base item dictionary
        item = {
            'text': text,
            'labels': torch.tensor(label, dtype=torch.long)
        }

        # Include metadata if present
        if self.source_datasets:
            item['source_dataset'] = self.source_datasets[idx]
        if self.source_models:
            item['source_model'] = self.source_models[idx]
        if self.filenames:
            item['filename'] = self.filenames[idx]

        # Tokenize if a tokenizer was provided
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                text,
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            # Remove batch dimension added by return_tensors='pt'
            item['input_ids'] = encoding['input_ids'].squeeze(0)
            item['attention_mask'] = encoding['attention_mask'].squeeze(0)
            # Some models use token_type_ids
            if 'token_type_ids' in encoding:
                item['token_type_ids'] = encoding['token_type_ids'].squeeze(0)

        return item
