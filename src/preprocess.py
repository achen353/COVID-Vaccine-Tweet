import re
import unicodedata
import itertools
import spacy
import nltk
from bs4 import BeautifulSoup
import emoji
from contractions import CONTRACTION_MAP
from emoticons import EMOTICONS


nlp = spacy.load('en_core_web_md', parse=True, tag=True, entity=True)
tokenizer = nlp.tokenizer
stopword_list = nltk.corpus.stopwords.words('english')


# Preprocess data
def preprocess(doc, hyperlink_removal=True, html_stripping=True,
               hashtag_removal=False, punctuation_removal=True,
               text_lower_case=True, contraction_expansion=True,
               misspell_fix=True, emoticon_replacement=True,
               emoji_replacement=True, stopword_removal=True,
               special_char_pattern_removal=True, special_char_removal=True,
               remove_digits=True):
    if not doc:
        return ""

    # Remove hyperlinks
    if hyperlink_removal:
        doc = remove_hyperlink(doc)

    # Strip off HTML tags
    if html_stripping:
        doc = strip_html_tags(doc)

    # Remove hashtags
    if hashtag_removal:
        doc = remove_hashtag(doc)
    else:
        doc = re.sub("#|@", "", doc)

    # Remove accented characters
    if punctuation_removal:
        doc = remove_punctuation(doc)

    # Lowercase the text
    if text_lower_case:
        doc = doc.lower()

    # Remove extra newlines
    doc = re.sub(r'[\r|\n|\r\n]+', ' ', doc)

    # Expand contractions
    if contraction_expansion:
        doc = expand_contractions(doc)

    # Fix mis-spelled words
    if misspell_fix:
        doc = fix_misspelled(doc)

    # Replace emotions
    if emoticon_replacement:
        doc = replace_emoticon(doc)

    # Replace emoji
    if emoji_replacement:
        doc = replace_emoji(doc)

    # Remove stopwords
    if stopword_removal:
        doc = remove_stopwords(doc, is_lower_case=text_lower_case)

    # Remove special char patterns
    if special_char_pattern_removal:
        doc = remove_special_char_pattern(doc)

    # Remove accented characters
    if special_char_removal:
        doc = remove_special_characters(doc, remove_digits)

    # Remove extra whitespace
    doc = re.sub(' +', ' ', doc)
    doc = doc.lstrip().rstrip()

    return doc


# Remove hyperlink
def remove_hyperlink(text):
    text = re.sub(r'http\S+', '', text)
    # Also remove the word 'href'
    text = re.sub(r'(href)|(href )', '', text)
    return text


# Strip HTML tags
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()
    return stripped_text


# Remove hashtag
def remove_hashtag(text):
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)", " ", text).split())
    return text


# Remove punctuation
def remove_punctuation(text):
    text = ' '.join(re.sub("[\.\,\!\?\:\;\-\=]", " ", text).split())
    return text


# Expand contractions as listed out in contractions.py
def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


# Fix mis-spelled words
def fix_misspelled(text):
    text = ''.join(''.join(s)[:2] for _, s in itertools.groupby(text))
    return text


# Replace emoticon
def replace_emoticon(text):
    words = text.split()
    reformed = [EMOTICONS[word] if word in EMOTICONS else word for word in words]
    text = " ".join(reformed)
    return text


# Replace emoji
def replace_emoji(text):
    text = emoji.demojize(text)
    text = text.replace(":", " ")
    text = ' '.join(text.split())
    return text


# Remove stop words
def remove_stopwords(text, is_lower_case=False):
    tokens = tokenizer(text)
    tokens = [token.orth_ for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


# Remove special char pattern
def remove_special_char_pattern(text):
    text = re.sub("([{.(-)!}])", "", text)
    return text


# Remove special characters
def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text
