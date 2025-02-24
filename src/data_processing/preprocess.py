import os
import re
import glob
import tarfile
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
from typing import Optional

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(console_handler)

def load_all_data(raw_folder: str) -> pd.DataFrame:
    """
    Load all data files from the raw folder and combine them efficiently.
    Handles CSV files and tar.bz2 files, minimizing memory usage with chunks.
    """
    df_list = []
    
    # Process CSV files in chunks to handle large files efficiently
    csv_files = glob.glob(os.path.join(raw_folder, "*.csv"))
    logger.info(f"Found {len(csv_files)} CSV file(s) in {raw_folder}")

    for file in csv_files:
        base = os.path.basename(file)
        logger.info(f"Processing CSV file: {file}")
        try:
            if "SMSSpamCollection" in base:
                logger.debug(f"Detected SMSSpamCollection format in {base}")
                chunk_size = 10000
                total_rows = sum(1 for _ in pd.read_csv(file, encoding='latin-1', sep="\t", header=None, chunksize=chunk_size))
                for chunk in pd.read_csv(file, encoding='latin-1', sep="\t", header=None, chunksize=chunk_size):
                    chunk.columns = ['label', 'message']
                    df_list.append(chunk)
                logger.debug(f"Loaded {base} - Processed in chunks, total rows: {total_rows}")
            else:
                df = pd.read_csv(file, encoding='latin-1', dtype={'message': str, 'label': str}, nrows=5)
                if 'v1' in df.columns and 'v2' in df.columns:
                    chunk_size = 10000
                    df = pd.read_csv(file, encoding='latin-1', usecols=['v1', 'v2'], dtype={'v1': str, 'v2': str}, chunksize=chunk_size)
                    for chunk in df:
                        chunk.columns = ['label', 'message']
                        df_list.append(chunk)
                    logger.debug(f"{base} - Converted columns v1, v2 to label, message. Total rows estimated.")
                elif 'label' in df.columns and 'message' in df.columns:
                    chunk_size = 10000
                    df = pd.read_csv(file, encoding='latin-1', usecols=['label', 'message'], dtype={'label': str, 'message': str}, chunksize=chunk_size)
                    for chunk in df:
                        df_list.append(chunk)
                    logger.debug(f"{base} - Using columns label, message. Total rows estimated.")
                elif 'label' in df.columns and 'text' in df.columns:
                    chunk_size = 10000
                    df = pd.read_csv(file, encoding='latin-1', usecols=['label', 'text'], dtype={'label': str, 'text': str}, chunksize=chunk_size)
                    for chunk in df:
                        chunk.columns = ['label', 'message']
                        df_list.append(chunk)
                    logger.debug(f"{base} - Converted column text to message. Total rows estimated.")
                else:
                    logger.warning(f"{file} does not have expected columns. Skipping.")
                    continue
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")
            continue

    # Process tar.bz2 files efficiently, streaming content without loading all at once
    tar_files = glob.glob(os.path.join(raw_folder, "*.tar.bz2"))
    logger.info(f"Found {len(tar_files)} tar.bz2 file(s) in {raw_folder}")

    for file in tar_files:
        logger.info(f"Processing tar.bz2 file: {file}")
        try:
            with tarfile.open(file, 'r:bz2') as tar:
                for member in tar.getmembers():
                    if member.isfile():
                        with tar.extractfile(member) as f:
                            if f is not None:
                                content = f.read().decode('latin-1', errors='ignore')
                                content = re.sub(r'\s+', ' ', content).strip()
                                temp_df = pd.DataFrame({'label': ['ham'], 'message': [content]})
                                df_list.append(temp_df)
                                logger.debug(f"Loaded file {member.name} from {os.path.basename(file)} with {len(content)} characters")
        except Exception as e:
            logger.error(f"Error processing {file}: {e}")

    if not df_list:
        raise ValueError("No valid data files found after processing.")

    # Concatenate efficiently with chunks and deduplicate
    df = pd.concat(df_list, ignore_index=True)
    logger.info(f"Combined dataframe shape (before deduplication and cleaning): {df.shape}")

    # Remove empty messages and deduplicate
    df = df[df['message'].str.strip().ne('')].drop_duplicates(subset=['message'], keep='first')
    logger.info(f"Combined dataframe shape (after deduplication and removing empties): {df.shape}")

    return df

def optionally_load_feedback(feedback_path: str) -> Optional[pd.DataFrame]:
    """
    Optionally load user feedback data (if the file exists) efficiently.
    Format in feedback.csv: [label, message, timestamp]
    We'll rename to [label, message] for consistency (drop timestamp).
    Preserves feedback.csv by backing it up and restoring it.
    """
    if not os.path.exists(feedback_path):
        logger.info("No feedback.csv found. Skipping feedback data.")
        return None

    feedback_backup = feedback_path.replace('.csv', '_backup.csv')
    try:
        # Backup feedback.csv before processing
        if os.path.exists(feedback_path):
            df = pd.read_csv(feedback_path, header=None, names=['label', 'message', 'timestamp'], dtype={'label': str, 'message': str})
            df.to_csv(feedback_backup, index=False, header=False)
            logger.info("Backed up feedback.csv to feedback_backup.csv")

        chunk_size = 10000
        df_list = []
        for chunk in pd.read_csv(feedback_path, header=None, names=['label', 'message', 'timestamp'], dtype={'label': str, 'message': str}, chunksize=chunk_size, encoding='utf-8'):
            chunk['message'] = chunk['message'].fillna('').astype(str).str.strip()
            chunk['label'] = chunk['label'].fillna('ham').astype(str).str.strip()  # Default to 'ham' for invalid labels
            chunk = chunk[chunk['message'].str.strip().ne('')]
            df_list.append(chunk[['label', 'message']])
        if not df_list:
            logger.warning("No valid feedback data found after processing.")
            # Restore backup if no valid data
            if os.path.exists(feedback_backup):
                pd.read_csv(feedback_backup, header=None, names=['label', 'message', 'timestamp'], dtype={'label': str, 'message': str}).to_csv(feedback_path, index=False, header=False)
                os.remove(feedback_backup)
                logger.info("Restored feedback.csv from backup due to no valid data")
            return None
        feedback_df = pd.concat(df_list, ignore_index=True).drop_duplicates(subset=['message'], keep='first')
        logger.info(f"Loaded feedback data with shape: {feedback_df.shape}")
        
        # Restore original feedback.csv after processing to preserve submissions
        if os.path.exists(feedback_backup):
            pd.read_csv(feedback_backup, header=None, names=['label', 'message', 'timestamp'], dtype={'label': str, 'message': str}).to_csv(feedback_path, index=False, header=False)
            os.remove(feedback_backup)
            logger.info("Restored feedback.csv from backup after processing")
        return feedback_df
    except Exception as e:
        logger.error(f"Could not read feedback.csv properly: {e}")
        # Restore backup if processing fails
        if os.path.exists(feedback_backup):
            pd.read_csv(feedback_backup, header=None, names=['label', 'message', 'timestamp'], dtype={'label': str, 'message': str}).to_csv(feedback_path, index=False, header=False)
            os.remove(feedback_backup)
            logger.info("Restored feedback.csv from backup after error")
        return None

def clean_text(text: str) -> str:
    """
    Clean text efficiently, retaining numbers and basic punctuation for context.
    """
    pattern = re.compile(r'[^a-zA-Z0-9\s.,!?-]')
    return pattern.sub('', text.lower()).strip()

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text cleaning to the dataset efficiently using vectorized operations.
    """
    logger.info(f"Starting text preprocessing on dataframe with shape: {df.shape}")
    
    # Clean and validate labels
    valid_labels = {'spam', 'ham'}
    df['label'] = df['label'].str.strip().str.lower()
    df['label'] = df['label'].apply(lambda x: x if x in valid_labels else 'ham')
    
    # Clean messages
    df['message'] = df['message'].apply(clean_text)
    logger.info("Completed text preprocessing.")
    return df

def split_and_save(df: pd.DataFrame, output_dir: str) -> None:
    """
    Split the dataset into training and testing sets and save them efficiently as .gz files.
    Does not modify feedback.csv or related files.
    """
    logger.info("Splitting data into train and test sets.")
    
    # Ensure labels are valid for stratification
    if not pd.api.types.is_categorical_dtype(df['label']) and df['label'].dtype == 'object':
        df['label'] = df['label'].astype('category')
    
    X = df['message']
    y = df['label']
    
    if len(y.unique()) < 2:
        raise ValueError("Dataset must have at least 2 unique labels for stratification.")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    train_df = pd.DataFrame({'message': X_train, 'label': y_train})
    test_df = pd.DataFrame({'message': X_test, 'label': y_test})
    
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, 'train.csv.gz')  # Explicitly use .gz extension
    test_path = os.path.join(output_dir, 'test.csv.gz')    # Explicitly use .gz extension
    
    train_df.to_csv(train_path, index=False, compression='gzip')
    test_df.to_csv(test_path, index=False, compression='gzip')
    logger.info(f"Preprocessed data saved to (compressed):")
    logger.info(f"       Training data: {train_path}")
    logger.info(f"       Testing data:  {test_path}")
    logger.debug("No modifications made to feedback.csv or related files during splitting")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, '..', '..')
    raw_folder = os.path.join(base_dir, 'data', 'raw')
    output_dir = os.path.join(base_dir, 'data', 'processed')
    feedback_path = os.path.join(base_dir, 'data', 'raw', 'feedback.csv')
    
    logger.info(f"Loading data from folder: {raw_folder}")
    df_main = load_all_data(raw_folder)
    logger.info("Data loaded. Label distribution (main dataset):")
    logger.info(df_main['label'].value_counts())
    
    df_feedback = optionally_load_feedback(feedback_path)
    if df_feedback is not None:
        logger.info("Merging main dataset with feedback data...")
        df_main = pd.concat([df_main, df_feedback], ignore_index=True)
        logger.info("New combined shape after feedback merge: %s", df_main.shape)
        logger.info("Label distribution (after feedback merge):")
        logger.info(df_main['label'].value_counts())
    
    df_main = preprocess_data(df_main)
    split_and_save(df_main, output_dir)