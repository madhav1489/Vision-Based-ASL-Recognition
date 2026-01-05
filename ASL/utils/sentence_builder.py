import os

WORDS_FILE = os.path.join(os.path.dirname(__file__), "words.txt")

DEFAULT_WORDS = [
    "hello", "help", "hi", "how",
    "boy", "book", "ball", "baby",
    "apple", "about", "after",
    "cat", "car", "come",
    "dog", "day",
    "yes", "you", "your",
    "good", "go", "great",
    "zoo"
]

def load_words():
    if os.path.exists(WORDS_FILE):
        with open(WORDS_FILE, "r") as f:
            return [w.strip().lower() for w in f if w.strip()]
    return DEFAULT_WORDS


WORDS = load_words()


def get_suggestions(sentence, max_results=4):
    if not sentence:
        return []

    last_word = sentence.lower().split(" ")[-1]
    if not last_word:
        return []

    matches = [w for w in WORDS if w.startswith(last_word) and w != last_word]
    return matches[:max_results]
