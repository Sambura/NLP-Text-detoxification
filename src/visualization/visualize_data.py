import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

try:
    from ..data.load_data import load_data, flatten_data
except ImportError:
    import sys
    if '.' not in sys.path: sys.path.append('.')
    from src.data.load_data import load_data, flatten_data

def get_lengths(df):
    flat = flatten_data(df)
    tokens = flat['text'].str.split('[ ?.!;,-]')
    return tokens.apply(len)

def plot_toxicity_scores(df, bins=80, export_path=None):
    plt.figure(figsize=(10, 7))
    ax = df[:]['trn_tox'].plot(kind='hist', bins=bins, label='Translations')
    df[:]['ref_tox'].plot(kind='hist', bins=bins, ax=ax, label='References')
    plt.xlabel('Toxicity score')
    ax.set_ylim(0, 80000)
    plt.title('Toxicity scores distribution')
    plt.legend(loc='upper center')

    if export_path is None:
        plt.show()
    else:
        plt.savefig(export_path, bbox_inches='tight')

def plot_lengths(lengths, bins=80, export_path=None):
    plt.figure(figsize=(10, 7))
    ax1 = lengths.plot(kind='hist', bins=80)
    ax2 = ax1.twinx()
    lengths.plot(kind='hist', bins=bins, ax=ax2, alpha=0.2, color='red')
    ax2.set_ylim(0, 20)
    plt.xlabel('Word count')
    plt.title('Word count distribution')

    if export_path is None:
        plt.show()
    else:
        plt.savefig(export_path, bbox_inches='tight')

def main(data_path, export_path):
    print('Loading data...')
    df = load_data(data_path, flatten=False)
    print('Calculating...')
    lengths = get_lengths(df)

    plot_toxicity_scores(df, export_path=os.path.join(export_path, 'toxicity_scores.png'))
    plot_lengths(lengths, export_path=os.path.join(export_path, 'sentence_lengths.png'))

if __name__ == '__main__':
    main('data/raw/filtered.tsv', 'reports/figures/')