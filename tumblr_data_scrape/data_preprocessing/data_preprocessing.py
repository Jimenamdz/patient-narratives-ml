###############################################################################
#  IMPORTS
###############################################################################

import os
import re
import time
import json
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
    r"(?i)^"
    # Exclude remission/cure signals
    r"(?!.*\b(?:remission|cancer\s*free|survivor)\b)"
    # Exclude pet, fanfic, and non-self indicators (e.g. friend, loved one, family members)
    r"(?!.*\b(?:pet|cat|dog|fanfic|chapter\s*\d+|fictional|friend|loved\s+one|mother|mom|father|dad|sister|brother|spouse|wife|husband|partner|cousin|aunt|uncle|grandmother|grandfather|family|relative|teacher|colleague|neighbor|mum|pa)\b)"
    # Require an explicit self-diagnosis phrase followed by a mention of cancer
    r"(?=.*\b(?:I\s+(?:have|am)|I\s+was\s+diagnosed\s+with)\b.*\bcancer\b)"
    # Look for terminal indicators
    r".*?\b(?:stage\s*(?:4|iv|four)|terminal\s*(?:cancer|illness|diagnosis)|"
    r"metastatic\s*(?:cancer|disease)|palliative\s*(?:care|treatment)|"
    r"end\s*of\s*life|hospice|months?\s*(?:to\s*(?:live|die)|left\s*(?:to\s*live))|"
    r"no\s*(?:cur(?:e|able)|more\s*treatment)|incurable|"
    r"spreading\s*(?:to|through)|aggressive\s*(?:form|type))\b",
    re.IGNORECASE | re.DOTALL
)

# Revised Non-Terminal Regex
non_terminal_regex = re.compile(
    r"(?i)^"
    # Exclude third-person references (including possessives) for family, friends, etc.
    r"(?!.*\bmy\s+(?:mother|mom|father|dad|sister|brother|spouse|wife|husband|partner|friend|loved\s+one|cousin|aunt|uncle|grandmother|grandfather|family|relative|teacher|colleague|neighbor|mum|pa)(?:['’]s)?\b)"
    # Exclude pet and fanfiction markers
    r"(?!.*\b(?:pet|cat|dog|fanfic|chapter\s*\d+|fanfiction|fictional)\b)"
    # Require explicit self-reference with a clear diagnosis phrase plus mention of cancer
    r"(?=.*\b(?:I\s+(?:have|am)|I\s+was\s+diagnosed\s+with|my\s+cancer)\b.*\bcancer\b)"
    # Look for non-terminal indicators
    r".*?\b(?:stage\s*(?:0|1|2|3|i|ii|iii|one|two|three)|remission|benign|"
    r"cancer\s*(?:free|survivor)|early\s*(?:detection|diagnosis)|"
    r"responding\s*(?:well|positively)|curable|low\s*risk|localized|treatable|"
    r"undergoing\s*(?:chemo|radiation|treatment|surgery)|removing\s*(?:tumor|mass))\b",
    re.IGNORECASE | re.DOTALL
)

# Revised Self-Experience Regex
self_experience_regex = re.compile(
    r"(?i)^"
    # Exclude third-person references (family, friends, etc.)
    r"(?!.*\bmy\s+(?:mother|mom|father|dad|sister|brother|spouse|wife|husband|partner|friend|loved\s+one|cousin|aunt|uncle|grandmother|grandfather|family|relative|teacher|colleague|neighbor|mum|pa)(?:['’]s)?\b)"
    # Exclude pet and fanfiction markers
    r"(?!.*\b(?:pet|cat|dog|furry|fanfic|chapter\s*\d+|fanfiction|fictional)\b)"
    # Require an explicit self-diagnosis phrase followed by mention of cancer
    r"(?=.*\b(?:I\s+(?:have|am)|I\s+was\s+diagnosed\s+with)\b.*\bcancer\b)"
    r".+",
    re.IGNORECASE | re.DOTALL
)
###############################################################################
# Folder creation helper
###############################################################################
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

###############################################################################
# Fetch posts from Tumblr API
###############################################################################

SEEN_POSTS_FILE = "seen_posts.json"
LAST_TIMESTAMP_FILE = "last_timestamp.json"

# Load previously seen post IDs
def load_seen_posts():
    if os.path.exists(SEEN_POSTS_FILE):
        with open(SEEN_POSTS_FILE, "r") as f:
            return set(json.load(f))
    return set()

# Save updated seen post IDs
def save_seen_posts(seen_posts):
    with open(SEEN_POSTS_FILE, "w") as f:
        json.dump(list(seen_posts), f)

# Load last fetched timestamp
def load_last_timestamp():
    if os.path.exists(LAST_TIMESTAMP_FILE):
        with open(LAST_TIMESTAMP_FILE, "r") as f:
            return json.load(f)
    return {}

# Save last fetched timestamp
def save_last_timestamp(timestamps):
    with open(LAST_TIMESTAMP_FILE, "w") as f:
        json.dump(timestamps, f)

def fetch_tumblr_posts(tags, max_posts_per_tag=5000):
    all_posts = []
    limit = 20  # Max posts per request

    seen_post_ids = load_seen_posts()
    last_timestamps = load_last_timestamp()

    for tag in tags:
        tag_posts = []
        before = last_timestamps.get(tag, None)  # Start from last timestamp

        while len(tag_posts) < max_posts_per_tag:
            url = f"https://api.tumblr.com/v2/tagged?tag={tag}&api_key={API_KEY}&limit={limit}"
            if before:
                url += f"&before={before}"

            response = requests.get(url)
            time.sleep(1)  # Prevent hitting rate limits

            if response.status_code == 200:
                data = response.json()
                posts = data.get("response", [])

                if not posts:
                    print(f"⚠️ No more posts found for tag: {tag}")
                    break

                new_posts = []
                for post in posts:
                    post_id = post.get("id")
                    if post_id not in seen_post_ids:
                        seen_post_ids.add(post_id)
                        new_posts.append(post)

                if not new_posts:  # If all posts are duplicates, stop fetching
                    print(f"⚠️ Duplicate loop detected for tag: {tag}. Stopping further requests.")
                    break

                tag_posts.extend(new_posts)
                before = posts[-1]["timestamp"]  # Save timestamp of oldest post

                print(f"✅ Retrieved {len(new_posts)} new posts for tag: {tag} (Total: {len(tag_posts)})")

            else:
                print(f"❌ Error fetching posts for tag '{tag}': {response.status_code}, Response: {response.text}")
                break

        all_posts.extend(tag_posts)
        if tag_posts:
            last_timestamps[tag] = tag_posts[-1]["timestamp"]  # Update timestamp for next run

    save_seen_posts(seen_post_ids)  # Save seen posts
    save_last_timestamp(last_timestamps)  # Save last timestamp per tag
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
# Utility: Print the total number of TXT files in each raw + preprocessed folder
###############################################################################
def print_file_counts_for_folder(folder_name):
    """
    Given a folder name (e.g. 'depression'), print
    how many .txt files are in the raw and preprocessed directories.
    """
    non_classified_path = f"tumblr_data_scrape/data/{folder_name}"
    raw_path = f"tumblr_data_scrape/data/tumblr_scraped_posts/{folder_name}"
    pre_path = f"tumblr_data_scrape/data/preprocessed_posts/{folder_name}"

    non_classified_count = 0
    raw_count = 0
    pre_count = 0

    # Count .txt files in pre-classified folder (if it exists)
    if os.path.exists(non_classified_path):
        non_classified_count = len([f for f in os.listdir(non_classified_path) if f.endswith('.txt')])

    # Count .txt files in raw folder (if it exists)
    if os.path.exists(raw_path):
        raw_count = len([f for f in os.listdir(raw_path) if f.endswith('.txt')])

    # Count .txt files in preprocessed folder (if it exists)
    if os.path.exists(pre_path):
        pre_count = len([f for f in os.listdir(pre_path) if f.endswith('.txt')])

    print(f"\nFolder: {folder_name}")
    print(f" - Non-classified files in '{non_classified_path}': {non_classified_count}")
    print(f" - Raw TXT files in '{raw_path}': {raw_count}")
    print(f" - Preprocessed TXT files in '{pre_path}': {pre_count}")

###############################################################################
# Fetchning and processing additional posts
############################################################################### 

def load_saved_posts(folder_path):
    posts = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, "r", encoding="utf-8") as file:
                lines = file.readlines()
            
            # Extract post details from the saved file
            post_data = {
                "id": filename.split("_")[1].replace(".txt", ""),
                "blog_name": lines[0].replace("Blog: ", "").strip() if len(lines) > 0 else "Unknown",
                "timestamp": int(datetime.strptime(lines[1].replace("Timestamp: ", "").strip(), "%Y-%m-%d %H:%M:%S").timestamp()) if len(lines) > 1 else 0,
                "tags": lines[2].replace("Tags: ", "").strip().split(", ") if len(lines) > 2 else [],
                "post_url": lines[3].replace("URL: ", "").strip() if len(lines) > 3 else "",
                "summary": lines[5].strip() if len(lines) > 5 else "",
                "body": "\n".join(lines[7:]).strip() if len(lines) > 7 else ""
            }
            posts.append(post_data)
    
    print(f"✅ Loaded {len(posts)} posts from '{folder_path}'")
    return posts


BLOG_LAST_TIMESTAMP_FILE = "blog_last_timestamp.json"

def load_blog_last_timestamp():
    if os.path.exists(BLOG_LAST_TIMESTAMP_FILE):
        with open(BLOG_LAST_TIMESTAMP_FILE, "r") as f:
            return json.load(f)
    return {}

def save_blog_last_timestamp(cache):
    with open(BLOG_LAST_TIMESTAMP_FILE, "w") as f:
        json.dump(cache, f)

       
def fetch_user_posts_after(blog_name, original_timestamp, max_posts=50):
    """
    Fetches posts from a specific blog that were created after the original_timestamp
    (or after the cached timestamp, if available) and within one month (30 days) afterwards.
    Stops once 'max_posts' additional posts have been collected.
    Updates the cache with the highest timestamp fetched.
    """
    # Load cached last timestamp for this blog (if any)
    blog_cache = load_blog_last_timestamp()
    blog_identifier = blog_name if blog_name.endswith('.tumblr.com') else f"{blog_name}.tumblr.com"
    # Use the cached timestamp if it's higher than the original
    effective_start = blog_cache.get(blog_identifier, original_timestamp)
    if effective_start < original_timestamp:
        effective_start = original_timestamp

    one_month_later = original_timestamp + 30 * 24 * 3600  # 30 days in seconds
    additional_posts = []
    offset = 0
    limit = 20
    iteration = 0
    max_fetched_timestamp = effective_start  # track highest timestamp seen

    while True:
        iteration += 1
        url = f"https://api.tumblr.com/v2/blog/{blog_identifier}/posts?api_key={API_KEY}&offset={offset}&limit={limit}"
        response = requests.get(url)

        # Handle rate limiting (HTTP 429)
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            retry_after = int(retry_after) if retry_after and retry_after.isdigit() else 60
            print(f"Rate limit hit for {blog_identifier}. Sleeping for {retry_after} seconds.")
            time.sleep(retry_after)
            continue

        if response.status_code != 200:
            print(f"❌ Error fetching posts for blog '{blog_identifier}': {response.status_code}")
            break

        time.sleep(1)  # Rate limit delay per request
        data = response.json()
        posts = data.get("response", {}).get("posts", [])
        if not posts:
            break

        total_posts = data.get("response", {}).get("total_posts", None)

        for post in posts:
            post_ts = post.get("timestamp", 0)
            # Only add posts that are newer than effective_start but within one month.
            if effective_start < post_ts <= one_month_later:
                additional_posts.append(post)
                if post_ts > max_fetched_timestamp:
                    max_fetched_timestamp = post_ts
                if len(additional_posts) >= max_posts:
                    blog_cache[blog_identifier] = max_fetched_timestamp
                    save_blog_last_timestamp(blog_cache)
                    return additional_posts
            elif post_ts <= effective_start:
                # Since posts are returned in reverse chronological order,
                # once we encounter a post older than effective_start, stop fetching.
                blog_cache[blog_identifier] = max_fetched_timestamp
                save_blog_last_timestamp(blog_cache)
                return additional_posts

        offset += limit

        if total_posts is not None and offset >= total_posts:
            break

        if iteration >= 100:
            print(f"Reached maximum iteration limit for blog '{blog_identifier}'.")
            break

    blog_cache[blog_identifier] = max_fetched_timestamp
    save_blog_last_timestamp(blog_cache)
    return additional_posts


def process_additional_posts(input_folder, raw_output_folder, preprocessed_output_folder, label_prefix):
    """
    Loads posts from the input folder (which can be either terminal or non-terminal),
    then for each post, fetches additional posts by the same user made up to one month
    after the original post. Both raw and preprocessed versions of these additional posts
    are saved in the specified output folders.
    """
    posts = load_saved_posts(input_folder)
    for post in posts:
        blog_name = post.get("blog_name", "Unknown")
        original_timestamp = post.get("timestamp", 0)
        if blog_name == "Unknown" or original_timestamp == 0:
            continue

        print(f"🔍 Fetching additional posts for user '{blog_name}' after timestamp {original_timestamp}...")
        additional_posts = fetch_user_posts_after(blog_name, original_timestamp)
        print(f"✅ Found {len(additional_posts)} additional posts for user '{blog_name}'.")

        for additional_post in additional_posts:
            # Save the raw additional post (if desired)
            save_post(additional_post, raw_output_folder, filename_prefix=f"{label_prefix}_additional")

###############################################################################
# Main execution
###############################################################################
def main():
    # Legacy tags (already scraped)
    # TAGS = [
    # "CancerJourney",
    # "cancer journey",
    # "LifeWithCancer",
    # "living with cancer",
    # "terminal cancer",
    # "remission",
    # "MyCancerStory",
    # "BreastCancerJourney",
    # "PancreaticCancerSucks",
    # "CancerSucks",
    # "CancerWarrior",
    # "cancer warrior",
    # "CancerChronicles",
    # ]
    TAGS = [
    "stage4cancer",
    "cancer stories",
    "BrainCancer",
    "my cancer story",
    "cancer fighter",
    "cancer",
    "thyroid cancer"
    "breast cancer",
    "hodgkins lymphoma",
    "fighting cancer",

    ]
    OUTPUT_FOLDER_TERMINAL = "tumblr_data_scrape/data/tumblr_scraped_posts/terminal"
    OUTPUT_FOLDER_NON_TERMINAL = "tumblr_data_scrape/data/tumblr_scraped_posts/non_terminal"
    PREPROCESSED_TERMINAL = "tumblr_data_scrape/data/preprocessed_posts/terminal"
    PREPROCESSED_NON_TERMINAL = "tumblr_data_scrape/data/preprocessed_posts/non_terminal"

    OUTPUT_FOLDER_TERMINAL_ADDITIONAL = "tumblr_data_scrape/data/tumblr_scraped_posts/terminal_additional"
    OUTPUT_FOLDER_NON_TERMINAL_ADDITIONAL = "tumblr_data_scrape/data/tumblr_scraped_posts/non_terminal_additional"
    PREPROCESSED_TERMINAL_ADDITIONAL = "tumblr_data_scrape/data/preprocessed_posts/terminal_additional"
    PREPROCESSED_NON_TERMINAL_ADDITIONAL = "tumblr_data_scrape/data/preprocessed_posts/non_terminal_additional"

    RAW_DATA = "tumblr_data_scrape/data/raw_data"

    create_folder(OUTPUT_FOLDER_TERMINAL)
    create_folder(OUTPUT_FOLDER_NON_TERMINAL)
    create_folder(PREPROCESSED_TERMINAL)
    create_folder(PREPROCESSED_NON_TERMINAL)

    create_folder(OUTPUT_FOLDER_TERMINAL_ADDITIONAL)
    create_folder(OUTPUT_FOLDER_NON_TERMINAL_ADDITIONAL)
    create_folder(PREPROCESSED_TERMINAL_ADDITIONAL)
    create_folder(PREPROCESSED_NON_TERMINAL_ADDITIONAL)

    create_folder(RAW_DATA)

    # print(f"Fetching Tumblr posts tagged '{TAGS}'...")
    # posts = fetch_tumblr_posts(TAGS)
    posts = load_saved_posts(RAW_DATA)

    for post in posts:
        #save_post(post, RAW_DATA, filename_prefix="raw")

        text_sources = [
                post.get("summary", ""),
                post.get("body", ""), 
                post.get("content", ""), 
                ]
        full_text = " ".join(text_sources).strip()

        if re.search(self_experience_regex, full_text):
            if re.search(terminal_regex, full_text):
                save_post(post, OUTPUT_FOLDER_TERMINAL, filename_prefix="terminal")
            elif re.search(non_terminal_regex, full_text):
                save_post(post, OUTPUT_FOLDER_NON_TERMINAL, filename_prefix="non_terminal")

    # Process additional posts for terminal posts
    print("=== Processing additional posts for terminal users ===")
    process_additional_posts(
        input_folder=OUTPUT_FOLDER_TERMINAL,
        raw_output_folder=OUTPUT_FOLDER_TERMINAL_ADDITIONAL,
        preprocessed_output_folder=PREPROCESSED_TERMINAL_ADDITIONAL,
        label_prefix="terminal"
    )

    # Process additional posts for non-terminal posts
    print("=== Processing additional posts for non-terminal users ===")
    process_additional_posts(
        input_folder=OUTPUT_FOLDER_NON_TERMINAL,
        raw_output_folder=OUTPUT_FOLDER_NON_TERMINAL_ADDITIONAL,
        preprocessed_output_folder=PREPROCESSED_NON_TERMINAL_ADDITIONAL,
        label_prefix="non_terminal"
    )

if __name__ == "__main__":
    main()
    print_file_counts_for_folder("raw_data")
    print_file_counts_for_folder("terminal")
    print_file_counts_for_folder("non_terminal")
    print_file_counts_for_folder("terminal_additional")
    print_file_counts_for_folder("non_terminal_additional")
