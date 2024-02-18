import sys
import pandas as pd
from tdc.utils import retrieve_label_name_list
from tdc.single_pred import CRISPROutcome


label_list = retrieve_label_name_list('Leenay')
data = CRISPROutcome(name = 'Leenay', label_name = label_list[0])

split = data.get_split(method = 'random')
print(split['test'].head(2))
print(split['train'].head(2))
# print(split['validation'].head(2))

def dataset_to_corpus(
    dataset:pd.DataFrame
    ):
    corpus = []
    for idx, row in dataset.iterrows():
        seq = row['GuideSeq']
        corpus.append([nucleotide for nucleotide in seq])
        if idx>3:
            print(corpus)
            sys.exit(1)


dataset_to_corpus(split['train'])