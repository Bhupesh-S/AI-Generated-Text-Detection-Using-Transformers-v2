import logging
from pathlib import Path

def setup_logger(log_file: str = "logs/data_loader.log") -> logging.Logger:
    """
    Setup and return a logger with file and console handlers.

    Args:
        log_file (str): The path to the log file.

    Returns:
        logging.Logger: Configured logger instance.
    """
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("DataLoaderLogger")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if function is called multiple times
    if not logger.handlers:
        # File handler
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.INFO)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger

def get_label_from_model(source_model: str) -> int:
    """
    Determine the class label based on the source model name.

    Args:
        source_model (str): The name of the folder indicating the text source (e.g., 'human', 'gpt').

    Returns:
        int: 0 for human-written text, 1 for AI-generated text.
    """
    # Define human folder names (case-insensitive)
    human_labels = ['human', 'human_written', 'real']

    if source_model.lower() in human_labels:
        return 0
    else:
        # Everything else (gpt, claude, gpt_prompt1, etc.) is considered AI-generated
        return 1
