import os
import random
import re
import shutil
import nltk
nltk.download('wordnet')
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# --------------------------------------------------------------------------
# Make sure you have run these once:
# nltk.download('wordnet')
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('omw-1.4')
# --------------------------------------------------------------------------

###############################################################################
# CONFIG
###############################################################################
RAW_ROOT_FOLDER = "data/tumblr_scraped_posts"       # Where your raw .txt files are (subfolders: depression, etc.)
PREPROCESSED_ROOT_FOLDER = "data/preprocessed_posts"  # Where your preprocessed .txt files are
AUGMENTED_ROOT_FOLDER = "data/augmented_posts"       # Where augmented files go, with matching subfolders
FILES_PER_POST_TYPE = 150                            # How many raw files to augment per subfolder
SWAP_FRACTION = 0.25                                 # Swap 25% of eligible words in each raw text

###############################################################################
# CREATE FOLDER IF NOT EXISTS
###############################################################################
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

###############################################################################
# PREPROCESSING FUNCTION
###############################################################################
def preprocess_text(text):
    """
    Lowercase, remove URLs, remove Reddit usernames, tokenize,
    remove stopwords, and apply stemming.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'u/\S+', '', text, flags=re.MULTILINE)

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]

    stemmer = PorterStemmer()
    tokens = [stemmer.stem(t) for t in tokens]

    return ' '.join(tokens)

###############################################################################
# GET SYNONYMS VIA WORDNET
###############################################################################
def get_synonyms(word):
    """Return a list of synonyms for a given word using WordNet."""
    synonyms = set()
    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().lower().replace('_', ' '))
    synonyms.discard(word.lower())
    return list(synonyms)

###############################################################################
# AUGMENT TEXT: SWAP ~25% OF ELIGIBLE WORDS
###############################################################################
def augment_text_25pct(raw_text, swap_fraction=0.25):
    """
    1) Whitespace-split the raw text.
    2) Identify candidate words (alphabetic, len>=3, not a stopword).
    3) Shuffle them, pick ~25%.
    4) Replace each with a random synonym if possible.
    5) Return the modified text.
    """
    tokens = raw_text.split()
    if not tokens:
        return raw_text

    stop = set(stopwords.words('english'))

    candidate_indices = []
    for i, w in enumerate(tokens):
        w_lower = w.lower()
        if w_lower.isalpha() and len(w_lower) >= 3 and w_lower not in stop:
            candidate_indices.append(i)

    num_to_swap = int(len(candidate_indices) * swap_fraction)
    if num_to_swap < 1:
        return raw_text

    random.shuffle(candidate_indices)
    indices_to_swap = candidate_indices[:num_to_swap]

    for idx in indices_to_swap:
        original_word = tokens[idx].lower()
        syns = get_synonyms(original_word)
        if syns:
            tokens[idx] = random.choice(syns)

    return " ".join(tokens)

###############################################################################
# COPY PREPROCESSED FILES INTO THE SAME AUGMENTED SUBFOLDER
###############################################################################
def copy_preprocessed_files_for_post_type(post_type):
    """
    Copy all .txt files from data/preprocessed_posts/<post_type> into
    data/augmented_posts/<post_type>.
    """
    pre_dir = os.path.join(PREPROCESSED_ROOT_FOLDER, post_type)
    aug_dir = os.path.join(AUGMENTED_ROOT_FOLDER, post_type)
    create_folder(aug_dir)  # Ensure it exists

    if not os.path.isdir(pre_dir):
        print(f"  No preprocessed folder found for '{post_type}' at '{pre_dir}'. Skipping copy.")
        return

    files_copied = 0
    for fname in os.listdir(pre_dir):
        if fname.endswith(".txt"):
            src = os.path.join(pre_dir, fname)
            dst = os.path.join(aug_dir, fname)
            shutil.copy2(src, dst)
            files_copied += 1

    print(f"  Copied {files_copied} preprocessed .txt files from '{pre_dir}' to '{aug_dir}'.")

###############################################################################
# MAIN
###############################################################################
def main():
    # 1) Create the top-level augmented folder
    create_folder(AUGMENTED_ROOT_FOLDER)

    # 2) Look for each post-type subfolder in RAW_ROOT_FOLDER
    if not os.path.exists(RAW_ROOT_FOLDER):
        print(f"ERROR: RAW_ROOT_FOLDER does not exist: {RAW_ROOT_FOLDER}")
        return

    # For each post type subfolder (e.g. depression, breastcancer, etc.)
    for post_type in sorted(os.listdir(RAW_ROOT_FOLDER)):
        subfolder_path = os.path.join(RAW_ROOT_FOLDER, post_type)
        if not os.path.isdir(subfolder_path):
            continue  # skip if not a directory

        # Gather raw .txt files in that subfolder
        raw_file_paths = []
        for filename in os.listdir(subfolder_path):
            if filename.endswith(".txt"):
                full_path = os.path.join(subfolder_path, filename)
                raw_file_paths.append(full_path)

        if not raw_file_paths:
            print(f"No .txt files found in '{subfolder_path}'. Skipping augmentation.")
            continue

        # Determine how many to augment
        total_files_to_augment = min(FILES_PER_POST_TYPE, len(raw_file_paths))

        # Make subfolder in augmented dir for this post type
        augmented_subfolder = os.path.join(AUGMENTED_ROOT_FOLDER, post_type)
        create_folder(augmented_subfolder)

        print(f"\n--- Post type: {post_type} ---")
        print(f" Found {len(raw_file_paths)} raw .txt files in '{subfolder_path}'.")
        print(f" Will augment {total_files_to_augment} of them (swap {int(SWAP_FRACTION*100)}% of eligible words).")
        print(f" Output => '{augmented_subfolder}'\n")

        # Randomly sample raw files to augment
        selected_paths = random.sample(raw_file_paths, total_files_to_augment)

        # Augment
        for i, file_path in enumerate(selected_paths, start=1):
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            # Separate metadata from the raw text
            text_start_index = None
            for idx, line in enumerate(lines):
                if line.strip() == "":
                    text_start_index = idx + 1
                    break

            if text_start_index is None:
                text_start_index = len(lines)

            metadata = lines[:text_start_index]
            text_lines = lines[text_start_index:]
            raw_text = " ".join(line.strip() for line in text_lines)

            # Perform synonym-swapping
            swapped_text = augment_text_25pct(raw_text, SWAP_FRACTION)

            # Preprocess the swapped text
            final_augmented_preprocessed = preprocess_text(swapped_text)

            # Build new filename, e.g. "aug_xyz123.txt"
            base_name = os.path.basename(file_path)
            new_filename = f"{base_name}_aug"
            new_path = os.path.join(augmented_subfolder, new_filename)

            # Write out metadata + final augmented text
            with open(new_path, "w", encoding="utf-8") as out_f:
                for m_line in metadata:
                    out_f.write(m_line)
                out_f.write("\n")
                out_f.write(final_augmented_preprocessed)

            if i % 25 == 0 or i == total_files_to_augment:
                print(f"  [SynAugment:{post_type}] Created {i}/{total_files_to_augment} augmented files...")

        # After augmentation, copy the preprocessed files for this same post type into augmented folder
        copy_preprocessed_files_for_post_type(post_type)

    print("\nAugmentation + copy complete for all post types!")
    print(f"Check in '{AUGMENTED_ROOT_FOLDER}' subfolders for both 'aug_' files and original preprocessed files.\n")


if __name__ == "__main__":
    main()

    print("\nAll done!")