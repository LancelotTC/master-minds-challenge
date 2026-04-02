import os

def get_files_since_date(data_dir: str, saved_date: str) -> list:
    """
    Scans the data directory for parquet files >= the saved_date.
    Assumes files are named exactly like 'gdelt_YYYYMMDD.parquet'.
    """
    valid_files = []
    
    for filename in os.listdir(data_dir):
        if filename.startswith("gdelt_") and filename.endswith(".parquet"):
            
            file_date = filename.replace("gdelt_", "").replace(".parquet", "")
            
            if file_date >= saved_date:
                full_path = os.path.join(data_dir, filename)
                valid_files.append(full_path)
    
    return sorted(valid_files)