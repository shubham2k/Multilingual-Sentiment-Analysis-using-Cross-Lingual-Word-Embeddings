import os
import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
import pandas as pd
import gensim
from argparse import ArgumentParser

def parse_dict(dpath):
    pairs = []
    for line in open(dpath,encoding="utf-8"):
        cols = line.strip().split('\t')
        if len(cols) == 2:
            weng,wwel = cols
            pairs.append((weng,wwel))
    return pairs

if __name__ == '__main__':

    parser = ArgumentParser()
    parser.add_argument('-d','--dict', help='Dictionary', required=True)
    parser.add_argument('-sv','--source-vocab', help='Source vocab', required=True)
    parser.add_argument('-tv','--target-vocab', help='Target vocab', required=True)
    parser.add_argument('-o','--output-folder', help='Directory to store train and test dictionaries', required=True)

    args = parser.parse_args()

    pairs = parse_dict(args.dict)

    print('Loading source vocab')
    source_vocab = set([line.strip() for line in open(args.source_vocab,encoding="utf-8")])
    print(f'Source vocab has {len(source_vocab)} words')
    print('---')
    print('Loading target vocab')
    target_vocab = set([line.strip() for line in open(args.target_vocab,encoding="utf-8")])
    print(f'Target vocab has {len(target_vocab)} words')

    # Split dictionary based on predefined vocab

    fpairs = []
    i = 0
    for srcw,tgtw in pairs:
        if srcw in source_vocab and tgtw in target_vocab:
            fpairs.append((srcw,tgtw))
        i += 1
        if i % 1000 == 0:
            print(f'Loading dictionary - Done {i} of {len(pairs)}')

    print(f'Loaded {len(fpairs)} entries from the original dictionary, which had: {len(pairs)}')

    fpairs = list(set(fpairs))

    fdf = pd.DataFrame(fpairs)
    fdf.columns = ['source', 'target']
    train_src, test_src, train_tgt, test_tgt = train_test_split(fdf.source, fdf.target, test_size = 0.2, shuffle=True)

    # save train dictionary
    out_train = pd.DataFrame(columns=['source', 'target'])
    out_train.source = train_src
    out_train.target = train_tgt
    out_train.to_csv(os.path.join(args.output_folder,'train_dict.csv'),
                     index=False, sep=' '
                    )
    # save test dictionary
    out_test = pd.DataFrame(columns=['source', 'target'])
    out_test.source = test_src
    out_test.target = test_tgt
    out_test.to_csv(os.path.join(args.output_folder,'test_dict.csv'),
                     index=False,sep=' '
                    )

