###############################################################################
#  IMPORTS
###############################################################################

import os
import re
import time
from datetime import datetime
import requests
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Configure Tumblr API credentials
API_KEY = "QsYWUpTvzr3OYJ4lGxnaRIxjbUXfx7JRMLhrJO6Oe1tyUr3JOQ"

# Define regex patterns for terminal and non-terminal cancer cases
terminal_regex = re.compile(
    r"(?i)\b(stage\s*(4|IV|four)|terminal\s*(cancer|illness|diagnosis)|metastatic\s*(cancer|disease)|palliative\s*(care|treatment)|end\s*of\s*life|hospice|months\s*(to|left\s*(to|live))|no\s*(cur(e|able)|more\s*treatment)|incurable|spreading\s*(to|through)|aggressive\s*(form|type))\b",
    re.IGNORECASE | re.DOTALL
)

non_terminal_regex = re.compile(
    r"(?i)\b(stage\s*(0|1|2|3|I|II|III|one|two|three)|remission|benign|cancer\s*(free|survivor)|early\s*(detection|diagnosis)|responding\s*(well|positively)|curable|low\s*risk|localized|treatable|undergoing\s*(chemo|radiation|treatment|surgery)|removing\s*(tumor|mass))\b",
    re.IGNORECASE | re.DOTALL
)
###############################################################################
# Preprocessing function
###############################################################################


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # Remove Markdown images
    text = re.sub(r'\[.*?\]\(.*?\)', '', text)  # Remove Markdown links
    text = re.sub(r'\[video\]', '', text)  # Remove embedded video tags
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)

###############################################################################
# Folder creation helper
###############################################################################
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

###############################################################################
# Fetch posts from Tumblr API
###############################################################################
def fetch_tumblr_posts(tags, max_posts_per_tag=2000):
    all_posts = []
    limit = 20
    for tag in tags:
        offset = 0
        tag_posts = []
        while len(tag_posts) < max_posts_per_tag:
            url = f"https://api.tumblr.com/v2/tagged?tag={tag}&api_key={API_KEY}&limit={limit}&offset={offset}"
            response = requests.get(url)
            #time.sleep(1)
            if response.status_code == 200:
                data = response.json()
                posts = data.get("response", [])
                if len(posts) < 20:
                    print(f"⚠️ No more posts found for tag: {tag}")
                    break
                tag_posts.extend(posts)
                offset += limit
                print(f"✅ Retrieved {len(posts)} posts for tag: {tag} (Total: {len(tag_posts)})")
            else:
                print(f"❌ Error fetching posts for tag '{tag}': {response.status_code}, Response: {response.text}")
                break
        all_posts.extend(tag_posts)
    print(f"Total posts retrieved across all tags: {len(all_posts)}")
    return all_posts

###############################################################################
# Save the raw scraped post
###############################################################################
def save_post(post, output_folder, filename_prefix):
    filename = f"{filename_prefix}_{post['id']}.txt"
    filepath = os.path.join(output_folder, filename)
    try:
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(f"Blog: {post.get('blog_name', 'Unknown')}\n")
            file.write(f"Timestamp: {datetime.utcfromtimestamp(post.get('timestamp', 0))}\n")
            file.write(f"Tags: {', '.join(post.get('tags', []))}\n")
            file.write(f"URL: {post.get('post_url', '')}\n\n")
            file.write(post.get('summary', ''))
            file.write("\n\n")
            file.write(post.get('body', ''))
        print(f"✅ Post saved: {filepath}")
    except Exception as e:
        print(f"⚠️ Error saving post {post['id']}: {e}")

###############################################################################
# Save the preprocessed post
###############################################################################
def save_preprocessed_post(post, output_folder, filename_prefix):
    preprocessed_text = preprocess_text(post.get('summary', '') + " " + post.get('body', ''))
    filename = f"{filename_prefix}_{post['id']}.txt"
    filepath = os.path.join(output_folder, filename)
    try:
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(preprocessed_text)
    except Exception as e:
        print(f"Error saving preprocessed post {post['id']}: {e}")

###############################################################################
# Fetch additional posts from the same user (within 1 month, across blogs)
###############################################################################
def fetch_user_posts(blog_identifier, start_time, end_time, limit=50):
    all_posts = []
    url = f"https://api.tumblr.com/v2/blog/{blog_identifier}/posts?api_key={API_KEY}&limit={limit}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        posts = data.get("response", {}).get("posts", [])
        for post in posts:
            post_timestamp = post.get("timestamp", 0)
            if start_time <= post_timestamp <= end_time:
                all_posts.append(post)
        return all_posts
    else:
        print(f"Error fetching user posts: {response.status_code}, Response: {response.text}")
        return []

def fetch_timeline_posts(blog_identifier, reference_timestamp, limit=50):
    one_month = 30 * 24 * 60 * 60
    seen_posts = set()
    try:
        before_posts = fetch_user_posts(blog_identifier, reference_timestamp - one_month, reference_timestamp, limit)
        after_posts = fetch_user_posts(blog_identifier, reference_timestamp, reference_timestamp + one_month, limit)
        timeline_posts = before_posts + after_posts
        return [p for p in timeline_posts if p['id'] not in seen_posts and not seen_posts.add(p['id'])]
    except Exception as e:
        print(f"⚠️ Error fetching timeline posts for {blog_identifier}: {e}")
        return []
###############################################################################
# Utility: Print the total number of TXT files in each raw + preprocessed folder
###############################################################################
def print_file_counts_for_folder(folder_name):
    """
    Given a folder name (e.g. 'depression'), print
    how many .txt files are in the raw and preprocessed directories.
    """
    raw_path = f"data/tumblr_scraped_posts/{folder_name}"
    pre_path = f"data/preprocessed_posts/{folder_name}"

    raw_count = 0
    pre_count = 0

    # Count .txt files in raw folder (if it exists)
    if os.path.exists(raw_path):
        raw_count = len([f for f in os.listdir(raw_path) if f.endswith('.txt')])

    # Count .txt files in preprocessed folder (if it exists)
    if os.path.exists(pre_path):
        pre_count = len([f for f in os.listdir(pre_path) if f.endswith('.txt')])

    print(f"\nFolder: {folder_name}")
    print(f" - Raw TXT files in '{raw_path}': {raw_count}")
    print(f" - Preprocessed TXT files in '{pre_path}': {pre_count}")
###############################################################################
# Main execution
###############################################################################
def main():
    TAGS = [
    "CancerJourney",
    "CancerSurvivor",
    "LifeWithCancer",
    "CancerDiary",
    "CancerSupport",
    "LivingWithCancer",
    "CancerAwareness",
    "MyCancerStory",
    "ChemoLife",
    "Stage4Cancer",
    "MetastaticCancer",
    "PalliativeCare",
    "CancerFighter",
    "CancerTreatment",
    "Oncology"
    ]
    OUTPUT_FOLDER_TERMINAL = "data/tumblr_scraped_posts/terminal"
    OUTPUT_FOLDER_NON_TERMINAL = "data/tumblr_scraped_posts/non_terminal"
    PREPROCESSED_TERMINAL = "data/preprocessed_posts/terminal"
    PREPROCESSED_NON_TERMINAL = "data/preprocessed_posts/non_terminal"
    create_folder(OUTPUT_FOLDER_TERMINAL)
    create_folder(OUTPUT_FOLDER_NON_TERMINAL)
    create_folder(PREPROCESSED_TERMINAL)
    create_folder(PREPROCESSED_NON_TERMINAL)

    print(f"Fetching Tumblr posts tagged '{TAGS}'...")
    posts = fetch_tumblr_posts(TAGS)

    for post in posts:
        text_sources = [
            post.get("summary", ""), post.get("body", ""), post.get("content", ""), 
            post.get("trail", [{}])[0].get("content_raw", "")
        ]
        full_text = " ".join(text_sources).strip()
        #full_text = preprocess_text(full_text)
        
        if re.search(terminal_regex, full_text):
            save_post(post, OUTPUT_FOLDER_TERMINAL, filename_prefix="terminal")
            save_preprocessed_post(post, PREPROCESSED_TERMINAL, filename_prefix="terminal")
        elif re.search(non_terminal_regex, full_text):
            save_post(post, OUTPUT_FOLDER_NON_TERMINAL, filename_prefix="non_terminal")
            save_preprocessed_post(post, PREPROCESSED_NON_TERMINAL, filename_prefix="non_terminal")
        
if __name__ == "__main__":
    main()
    print_file_counts_for_folder("terminal")
    print_file_counts_for_folder("non_terminal")
