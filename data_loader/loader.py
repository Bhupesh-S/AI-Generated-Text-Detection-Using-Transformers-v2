import sys
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Add current directory to path to allow importing local modules
sys.path.append(str(Path(__file__).parent))

import utils
from dataset import AIGeneratedTextDataset

# Define Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "datasets" / "merged"
OUT_DIR = BASE_DIR / "datasets" / "processed"
OUT_CSV_PATH = OUT_DIR / "folder_dataset.csv"

# Initialize logger
logger = utils.setup_logger()

def create_dataframe_from_folders(data_dir: Path) -> pd.DataFrame:
    """
    Recursively traverse the dataset folders, read .txt files, and compile them into a DataFrame.

    Args:
        data_dir (Path): Base directory containing the datasets.

    Returns:
        pd.DataFrame: A DataFrame containing the texts, labels, and metadata.
    """
    logger.info(f"Starting to traverse directory: {data_dir}")

    if not data_dir.exists():
        logger.error(f"Directory {data_dir} does not exist.")
        raise FileNotFoundError(f"Directory {data_dir} does not exist.")

    # Gather all .txt files recursively
    txt_files = list(data_dir.rglob("*.txt"))
    logger.info(f"Found {len(txt_files)} text files.")

    records = []

    for file_path in tqdm(txt_files, desc="Reading text files"):
        try:
            # We assume the directory structure is: datasets/merged/<source_dataset>/<source_model>/<filename>.txt
            # Example: datasets/merged/essay/human/1.txt

            # The parent directory is the source_model (e.g., 'human', 'gpt', 'claude')
            source_model = file_path.parent.name

            # The parent of the parent is the source_dataset (e.g., 'essay', 'reuter', 'wp')
            source_dataset = file_path.parent.parent.name

            filename = file_path.name

            # Determine label
            label = utils.get_label_from_model(source_model)

            # Read text content
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                text_content = f.read().strip()

            # Skip empty files
            if not text_content:
                continue

            records.append({
                'text': text_content,
                'label': label,
                'source_dataset': source_dataset,
                'source_model': source_model,
                'filename': filename
            })

        except Exception as e:
            logger.warning(f"Failed to process file {file_path}. Error: {e}")

    df = pd.DataFrame(records)
    logger.info(f"Successfully processed {len(df)} non-empty text files.")
    return df

def print_dataset_statistics(df: pd.DataFrame):
    """
    Print comprehensive statistics about the dataset.
    """
    logger.info("Generating dataset statistics...")

    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)

    print(f"\nTotal Samples: {len(df)}")

    print("\nClass Distribution:")
    class_dist = df['label'].value_counts()
    print(f"  AI-Generated (1): {class_dist.get(1, 0)} ({(class_dist.get(1, 0)/len(df))*100:.2f}%)")
    print(f"  Human-Written (0): {class_dist.get(0, 0)} ({(class_dist.get(0, 0)/len(df))*100:.2f}%)")

    print("\nSource Dataset Distribution:")
    dataset_dist = df['source_dataset'].value_counts()
    for ds, count in dataset_dist.items():
        print(f"  {ds}: {count} samples")

    print("\nSource Model Distribution:")
    model_dist = df['source_model'].value_counts()
    for model, count in model_dist.items():
        print(f"  {model}: {count} samples")

    print("="*50 + "\n")

def split_dataset(df: pd.DataFrame) -> tuple:
    """
    Split the dataset into Train (70%), Valid (15%), Test (15%).

    Args:
        df (pd.DataFrame): The complete dataset.

    Returns:
        tuple: (train_df, valid_df, test_df)
    """
    logger.info("Splitting dataset into Train/Valid/Test (70/15/15)...")

    # Stratify split to maintain class ratio
    train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['label'])

    # Split the 30% temp into 15% valid and 15% test
    valid_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['label'])

    logger.info(f"Split completed: Train={len(train_df)}, Valid={len(valid_df)}, Test={len(test_df)}")
    return train_df, valid_df, test_df

def main():
    logger.info("--- Data Loader Execution Started ---")

    # 1. Create processed output directory
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 2. Traverse folders and create DataFrame
    df = create_dataframe_from_folders(DATA_DIR)

    if df.empty:
        logger.error("No valid text data found. Exiting.")
        print("No text data found in the specified directory.")
        return

    # 3. Print statistics
    print_dataset_statistics(df)

    # 4. Save compiled dataset to CSV
    logger.info(f"Saving compiled dataset to {OUT_CSV_PATH}")
    df.to_csv(OUT_CSV_PATH, index=False)
    print(f"Compiled dataset saved to {OUT_CSV_PATH}")

    # 5. Split Dataset
    train_df, valid_df, test_df = split_dataset(df)

    # Optional: Save split datasets
    train_df.to_csv(OUT_DIR / "train.csv", index=False)
    valid_df.to_csv(OUT_DIR / "valid.csv", index=False)
    test_df.to_csv(OUT_DIR / "test.csv", index=False)
    logger.info("Split datasets saved successfully.")

    # 6. Demonstrate Custom PyTorch Dataset usage
    logger.info("Instantiating custom PyTorch Dataset for the training set...")
    train_dataset = AIGeneratedTextDataset(train_df)

    print("\nPyTorch Dataset Sample (Train Set Index 0):")
    sample = train_dataset[0]
    print(f"  Filename: {sample.get('filename')}")
    print(f"  Source Model: {sample.get('source_model')}")
    print(f"  Label: {sample.get('labels').item()}")
    print(f"  Text Prefix: {sample.get('text')[:100]}...")

    logger.info("--- Data Loader Execution Completed ---")

if __name__ == "__main__":
    main()
