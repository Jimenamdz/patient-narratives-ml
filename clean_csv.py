#!/usr/bin/env python3

import re
import pandas as pd
from bs4 import BeautifulSoup

def parse_after_keyword(text_value, keyword):
    pattern = re.compile(rf"{keyword}:(.*)", re.IGNORECASE | re.DOTALL)
    match = pattern.search(str(text_value))
    return match.group(1).strip() if match else text_value

def strip_tumblr_headers(text_value):
    lines = text_value.splitlines()
    stripped_lines = [
        line for line in lines
        if not re.match(r"^\s*(Blog:|Timestamp:|Tags:|URL:)", line, flags=re.IGNORECASE)
    ]
    return " ".join(stripped_lines)

def parse_and_extract(text_value):
    parsed = parse_after_keyword(text_value, "Content")
    if parsed == text_value:
        parsed = parse_after_keyword(text_value, "URL")
    parsed = strip_tumblr_headers(parsed)
    return parsed

def remove_emojis(text):
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002600-\U000026FF"
        "\U00002700-\U000027BF"
        "\U000024C2-\U0001F251"
        "\U0001f926-\U0001f937"
        "\U00010000-\U0010ffff"
        "\u2640-\u2642"
        "\u2600-\u2B55"
        "\u200d"
        "\u23cf"
        "\u23e9"
        "\u231a"
        "\ufe0f"
        "\u3030"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r'', text)

def clean_text(raw_text):
    soup = BeautifulSoup(str(raw_text), "html.parser")
    text = soup.get_text(separator=" ")
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\[[^\]]*\]", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#[\w]+", "", text)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}", "", text)
    text = re.sub(r"\+?\d[\d\-\(\) ]+\d", "", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = remove_emojis(text)
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def valid_text(text):
    words = [word for word in text.split() if len(word) >= 3]
    return len(words) >= 5

def main():
    df = pd.read_csv("patient_test_data_with_source.csv", encoding="utf-8")
    df["text"] = df["text"].apply(parse_and_extract)
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].apply(valid_text)]
    df.drop_duplicates(subset="text", keep="first", inplace=True)
    df.to_csv("cleaned_patient_data.csv", index=False, encoding="utf-8")
    print("Cleaning complete. Results saved to cleaned_patient_data.csv")

if __name__ == "__main__":
    main()