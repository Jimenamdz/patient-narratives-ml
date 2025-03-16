#!/usr/bin/env python3

import re
import pandas as pd
from bs4 import BeautifulSoup

def parse_after_keyword(text_value, keyword):
    """
    Returns everything AFTER the first occurrence of `keyword:` in `text_value`.
    If not found, returns the original text_value unchanged.
    Matching is case-insensitive.
    """
    if not isinstance(text_value, str):
        text_value = str(text_value)
    # Build a regex like:  CONTENT:(.*) or URL:(.*) case-insensitive
    pattern = re.compile(rf"{keyword}:(.*)", re.IGNORECASE | re.DOTALL)
    match = pattern.search(text_value)
    if match:
        return match.group(1).strip()
    else:
        return text_value

def strip_tumblr_headers(text_value):
    """
    Removes lines that start with Blog:, Timestamp:, Tags:, or URL: (case-insensitive).
    Leaves everything else intact.
    """
    lines = text_value.splitlines()
    stripped_lines = []
    for line in lines:
        # If line begins with "Blog:", "Timestamp:", "Tags:", or "URL:", skip it
        if re.match(r"^\s*(Blog:|Timestamp:|Tags:|URL:)", line, flags=re.IGNORECASE):
            continue
        stripped_lines.append(line)
    # Join them back with a space or newline
    return " ".join(stripped_lines)

def parse_and_extract(text_value):
    """
    1. Try extracting everything AFTER 'Content:' (for Reddit).
    2. If 'Content:' not found, try extracting everything AFTER 'URL:' (for Tumblr).
    3. Remove lines that start with Blog:, Timestamp:, Tags:, URL: 
       (in case there's leftover Tumblr headers).
    """
    # First try 'Content:'
    parsed = parse_after_keyword(text_value, "Content")
    # If it didn't change, there's no 'Content:' => check 'URL:' next
    if parsed == text_value:
        parsed = parse_after_keyword(text_value, "URL")
    # Remove lines beginning with Blog:, Timestamp:, Tags:, or URL:
    parsed = strip_tumblr_headers(parsed)
    return parsed

def clean_text(raw_text):
    """
    Cleans the extracted text by:
    - Removing HTML (BeautifulSoup)
    - Removing URLs, placeholders [icon:...], @usernames, #hashtags
    - Removing emails, phone numbers
    - Merging excessive punctuation
    - Normalizing whitespace
    """
    if not isinstance(raw_text, str):
        raw_text = str(raw_text)

    # Remove HTML tags
    soup = BeautifulSoup(raw_text, "html.parser")
    text = soup.get_text(separator=" ")

    # Remove URLs (http, https, www)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)

    # Remove anything in square brackets [icon: ...], etc.
    text = re.sub(r"\[[^\]]*\]", "", text)

    # Remove @usernames, #hashtags
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)

    # Remove emails
    text = re.sub(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", "", text)

    # Remove phone numbers (+ optional brackets, hyphens, spaces)
    text = re.sub(r"\+?\d[\d\-\(\) ]+\d", "", text)

    # (Optional) remove leftover HTML tags if any remain
    text = re.sub(r"<[^>]+>", "", text)

    # Merge or remove repeated punctuation, e.g. "!!!" => "!"
    text = re.sub(r"[^\w\s]+(?=[^\w\s]+)", "", text)
    text = re.sub(r"[^\w\s]{2,}", "", text)

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text

def main():
    # Load your CSV
    df = pd.read_csv("patient_test_data_with_source.csv", encoding="utf-8")

    # 1) Extract the real post content (after 'Content:' or 'URL:', removing Tumblr headers)
    df["text"] = df["text"].apply(parse_and_extract)

    # 2) Clean the extracted text
    df["text"] = df["text"].apply(clean_text)

    # 3) Drop duplicates based on the final cleaned 'text'
    df.drop_duplicates(subset="text", keep="first", inplace=True)

    # 4) Save the result as a new CSV, preserving all columns
    df.to_csv("cleaned_patient_data.csv", index=False, encoding="utf-8")

    print("Cleaning complete. Results saved to cleaned_patient_data.csv")

if __name__ == "__main__":
    main()
