import pandas as pd
from tqdm.auto import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import os

from string import ascii_lowercase as ascii_lowercase_string

class StringCharactersKeeper:
  def __init__(self, keep):
    self.comp = dict((ord(c),c) for c in keep)
  def __getitem__(self, k):
    return self.comp.get(k, ' ')

__trn_lowercase_keeper = StringCharactersKeeper(ascii_lowercase_string)

def lower_text(text: str):
    return text.lower()

def remove_all_except_letters(text: str):
    return text.translate(__trn_lowercase_keeper)

def remove_multiple_spaces(text: str):
    return " ".join(text.split())

def tokenize_text(text: str) -> list[str]:
    return text.split()

def get_tokens(text):
    _lowered = lower_text(text)
    _only_letters = remove_all_except_letters(_lowered)
    return tokenize_text(_only_letters)

def load_dataframe_and_count_words(path, col1='reference', col2='translation', remove_contractions=True):
    df = pd.read_csv(path, sep='\t').astype(str)
    input_tokens = Counter()
    detox_tokens = Counter()

    for inp, det in tqdm(zip(df[col1], df[col2]), total=len(df.index)):
        input_tokens.update(get_tokens(inp))
        detox_tokens.update(get_tokens(det))
    
    if remove_contractions:
        delete_words = ['s', 'll', 're', 'm', 't']
    
    for word in delete_words:
        del input_tokens[word]
        del detox_tokens[word]

    return input_tokens, detox_tokens

def compute_differences(in_tokens, out_tokens, sort=True, separate=True):
    differences = []

    for word, count in in_tokens.items():
        differences.append((word, out_tokens.get(word, 0) - count))

    for word, count in out_tokens.items():
        if word not in out_tokens: differences.append((word, count))

    if sort: differences.sort(key=lambda x: abs(x[1]), reverse=True)

    if separate:
        added = [x for x in differences if x[1] > 0]
        removed = [x for x in differences if x[1] < 0]
        same = [x for x in differences if x[1] == 0]

        return added, removed, same

    return differences

def plot_words(word_list, top_n=25, color=None, title=None, export_path=None):
    words, diff_values = zip(*(word_list[:top_n]))

    plt.figure(figsize=(12, 6))
    plt.bar(words, diff_values, color=color)
    plt.ylabel('Difference')
    plt.title(title)
    plt.xticks(rotation=50, fontsize=8)
    plt.tight_layout() 

    if export_path is not None:
        plt.savefig(export_path, bbox_inches='tight')
        return

    plt.show()

def main(predictions_path, export_path, n=40):
    in_tokens, out_tokens = load_dataframe_and_count_words(predictions_path)
    added, removed, same = compute_differences(in_tokens, out_tokens)

    plot_words(
        added, 
        top_n=n, 
        color='teal', 
        title='Top words added during detoxification',
        export_path=os.path.join(export_path, 'added_words.png')
    )

    plot_words(
        removed, 
        top_n=n, 
        color='orange', 
        title='Top words removed during detoxification',
        export_path=os.path.join(export_path, 'removed_words.png')
    )

if __name__ == '__main__':
    main(
        predictions_path='data/predicted/predictions.tsv',
        export_path='reports/figures/'
    )