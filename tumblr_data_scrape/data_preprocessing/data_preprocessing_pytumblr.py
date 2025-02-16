import os
import re
import time
import pytumblr
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')

# Authenticate via pytumblr
client = pytumblr.TumblrRestClient(
    'QsYWUpTvzr3OYJ4lGxnaRIxjbUXfx7JRMLhrJO6Oe1tyUr3JOQ',
    'JAFOY5gjhuTraWDAdCNQy1mdZ3hIO3CBHjT0cuQqK21O5cm7eU',
    'QsYWUpTvzr3OYJ4lGxnaRIxjbUXfx7JRMLhrJO6Oe1tyUr3JOQ',
    'JAFOY5gjhuTraWDAdCNQy1mdZ3hIO3CBHjT0cuQqK21O5cm7eU'
)

# Regex patterns for classification
terminal_regex = re.compile(
    r"(?i)\b(stage\s*(4|IV|four)|terminal\s*(cancer|illness|diagnosis)|metastatic\s*(cancer|disease)|"
    r"palliative\s*(care|treatment)|end\s*of\s*life|hospice|months\s*(to|left\s*(to|live))|"
    r"no\s*(cur(e|able)|more\s*treatment)|incurable|spreading\s*(to|through)|"
    r"aggressive\s*(form|type))\b",
    re.IGNORECASE | re.DOTALL
)

non_terminal_regex = re.compile(
    r"(?i)\b(stage\s*(0|1|2|3|I|II|III|one|two|three)|remission|benign|cancer\s*(free|survivor)|"
    r"early\s*(detection|diagnosis)|responding\s*(well|positively)|curable|low\s*risk|localized|treatable|"
    r"undergoing\s*(chemo|radiation|treatment|surgery)|removing\s*(tumor|mass))\b",
    re.IGNORECASE | re.DOTALL
)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)  # Remove HTML tags
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)

def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)


def fetch_tumblr_posts(tags, max_posts_per_tag=1000):
    all_posts = []

    for tag in tags:
        tag_posts = []
        before = None  # Timestamp for pagination

        while len(tag_posts) < max_posts_per_tag:
            try:
                posts = client.tagged(tag, before=before) if before else client.tagged(tag)
                time.sleep(1)  # Respect API rate limits

                if not posts or len(posts) == 0:
                    print(f"⚠️ No more posts found for tag: {tag}")
                    break  # Stop fetching if no posts returned

                tag_posts.extend(posts)

                # Check if the last post has a 'timestamp'
                if "timestamp" in posts[-1]:
                    before = posts[-1]['timestamp']
                else:
                    print(f"⚠️ Last retrieved post for {tag} lacks a timestamp. Stopping pagination.")
                    break  # If no timestamp, we can't paginate

                print(f"✅ Retrieved {len(posts)} posts for tag: {tag} (Total: {len(tag_posts)})")

            except Exception as e:
                print(f"❌ Error fetching posts for tag '{tag}': {e}")
                break  # Stop on API failure

        all_posts.extend(tag_posts[:max_posts_per_tag])  # Limit to max_posts_per_tag per tag

    print(f"Total posts retrieved across all tags: {len(all_posts)}")
    return all_posts


def save_preprocessed_post(post, output_folder, filename_prefix):
    preprocessed_text = preprocess_text(post.get('summary', '') + " " + post.get('body', ''))
    filename = f"{filename_prefix}_{post['id']}.txt"
    filepath = os.path.join(output_folder, filename)
    try:
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(preprocessed_text)
    except Exception as e:
        print(f"Error saving preprocessed post {post['id']}: {e}")

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

def main():
    TAGS = [
        "CancerJourney", "CancerSurvivor", "LifeWithCancer", "CancerDiary", "CancerSupport",
        "LivingWithCancer", "CancerAwareness", "MyCancerStory", "ChemoLife", "Stage4Cancer",
        "MetastaticCancer", "PalliativeCare", "CancerFighter", "CancerTreatment", "Oncology"
    ]
    OUTPUT_TERMINAL = "data/tumblr_scraped_posts/terminal"
    OUTPUT_NON_TERMINAL = "data/tumblr_scraped_posts/non_terminal"
    PREPROCESSED_TERMINAL = "data/preprocessed_posts/terminal"
    PREPROCESSED_NON_TERMINAL = "data/preprocessed_posts/non_terminal"
    
    create_folder(OUTPUT_TERMINAL)
    create_folder(OUTPUT_NON_TERMINAL)
    create_folder(PREPROCESSED_TERMINAL)
    create_folder(PREPROCESSED_NON_TERMINAL)
    
    print("Fetching Tumblr posts...")
    posts = fetch_tumblr_posts(TAGS)
    
    for post in posts:
        text_sources = [
        post.get("summary", ""), 
        post.get("body", ""), 
        post.get("content", ""), 
        post["trail"][0]["content_raw"] if post.get("trail") and len(post["trail"]) > 0 else ""
        ]

        full_text = " ".join(text_sources).strip()
        
        if re.search(terminal_regex, full_text):
            save_post(post, OUTPUT_TERMINAL, filename_prefix="terminal")
            save_preprocessed_post(post, PREPROCESSED_TERMINAL, filename_prefix="terminal")
        elif re.search(non_terminal_regex, full_text):
            save_post(post, OUTPUT_NON_TERMINAL, filename_prefix="non_terminal")
            save_preprocessed_post(post, PREPROCESSED_NON_TERMINAL, filename_prefix="non_terminal")

if __name__ == "__main__":
    main()
    print_file_counts_for_folder("terminal")
    print_file_counts_for_folder("non_terminal")
