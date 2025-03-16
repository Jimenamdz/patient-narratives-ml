import os
import pandas as pd

def guess_labels_from_path(path):
    """
    Infer 'group' (e.g., 'terminal' or 'non-terminal') and 'source' ('tumblr' or 'reddit')
    by looking for keywords in the given file path.
    
    For example, if the path is something like:
       ".../tumblr_scraped_posts/terminal/some_file.txt"
    then the folder portion likely contains 'tumblr' and 'terminal'.
    """
    lower_path = path.lower()

    # Default / fallback
    group_label = "unknown_group"
    source_label = "unknown_source"

    # Detect "terminal" vs. "non_terminal"
    # If both appear in the path, the latter check will overwrite the earlier,
    # so if your paths contain both words for some reason, adjust accordingly.
    if "terminal" in lower_path:
        group_label = "terminal"
    if "non_terminal" in lower_path:
        group_label = "non-terminal"

    # Detect "tumblr" vs. "reddit"
    if "tumblr" in lower_path:
        source_label = "tumblr"
    if "reddit" in lower_path:
        source_label = "reddit"

    return group_label, source_label


def load_texts_recursively(base_folder):
    """
    Recursively walk all subfolders from 'base_folder' and look for .txt files.
    For each .txt file, call guess_labels_from_path() to determine 'group' & 'source'.
    
    Returns a list of dict objects:
       [
         { "group": ..., "text": ..., "source": ... },
         ...
       ]
    """
    rows = []
    
    for root, dirs, files in os.walk(base_folder):
        for filename in files:
            if filename.endswith(".txt"):
                full_path = os.path.join(root, filename)
                
                # Infer group / source from the path
                group_label, source_label = guess_labels_from_path(full_path)
                
                # Read the file content
                with open(full_path, "r", encoding="utf-8") as f:
                    text_content = f.read().strip()

                # Append a row (dict)
                rows.append({
                    "group": group_label,    # e.g. "terminal" or "non-terminal"
                    "text": text_content,    # entire text from the .txt file
                    "source": source_label   # e.g. "tumblr" or "reddit"
                })
    
    return rows


if __name__ == "__main__":
    # Choose the folder to search. "." means current directory (and all subfolders).
    base_folder = "."

    # Gather all .txt files recursively
    all_rows = load_texts_recursively(base_folder)

    # Build a DataFrame and save to CSV
    df = pd.DataFrame(all_rows, columns=["group", "text", "source"])
    output_csv = "patient_test_data_with_source.csv"
    df.to_csv(output_csv, index=False)

    print(f"Data merged into {output_csv} with columns: group, text, source.")
    print(f"Found {len(df)} records total.")
