import torch
import torch.nn as nn
from collections import Counter, OrderedDict
from torchtext.vocab import vocab
from typing import List
import pandas as pd
import sys
from tdc.utils import retrieve_label_name_list
from tdc.single_pred import Develop
from torchtext.vocab import vocab

def collate_batch(batch, vocab_map):
		label_list, processed_seqs, lengths = [], [],[]

		for _lab, _seq in batch:
			label_list.append(_lab)
			processed_seq= torch.tensor([vocab_map[x] for x in _seq],dtype = torch.int64)
			processed_seqs.append(processed_seq)
			lengths.append(processed_seq.size(0))
		
		label_list = torch.tensor(label_list,dtype = torch.float)
		lengths = torch.tensor(lengths)
		padded_text = nn.utils.rnn.pad_sequence(processed_seqs, batch_first = True)
		return padded_text, label_list, lengths


def make_vocab(
	corpus:List
	):
	

	token_counter = Counter()
	for _, sequence in corpus:
		token_counter.update(sequence)
	
	sorted_tokens = sorted(token_counter.items(), key= lambda x: x[1], reverse = True)
	sorted_tokens = OrderedDict(sorted_tokens)
	vocab_map = vocab(sorted_tokens)
	vocab_map.insert_token('<pad>',0)
	
	if len(token_counter)<21:
		# have we missed an amino? 
		vocab_map.insert_token('<unk>',1)



	
	return vocab_map
	


def process_regression_antibody(
	ab:str
	)->str:
	
	split_ab= ab[2:-2].split("'")
	heavy_chain = split_ab[0]
	light_chain = split_ab[-1]
	
	return "".join([heavy_chain,"&",light_chain])
	

def process_classification_antibody(
	ab:str
	)->str:
	
	heavy_chain, light_chain = ab[2:-2].split(",")
	heavy_chain = heavy_chain[:-1]
	light_chain = light_chain.strip()
	light_chain = light_chain[1:]
	return "".join([heavy_chain,"&", light_chain])

def build_corpus(
	data:pd.DataFrame,
	is_classif:bool = True
	)->List[str]:
	

	corpus = []

	for idx, row in data.iterrows():
		label = row['Y']

		if is_classif:
			antibody = process_classification_antibody(row['Antibody'])	
		else:
			antibody = process_regression_antibody(row['Antibody'])
		
		antibody = [x for x in antibody]
		
		corpus.append((label, antibody))


	return corpus


def make_classification_dataset(
	path:str = "../data/",
	name:str = "SAbDab_Chen"
	):
	

	dataset = Develop(path = path, name = name)
	data = dataset.get_data()
	

	k = 30
	train_data = dataset.get_split()['train']
	positive_samples = train_data[train_data['Y']==1]
	negative_samples = train_data[train_data['Y']==0]
	negative_samples = negative_samples.sample(positive_samples.shape[0])
	train_data = pd.concat([positive_samples.iloc[0:k,:],negative_samples.iloc[0:k,:]],axis=0)
	# train_data = pd.concat([positive_samples,negative_samples],axis=0)

	
	val_data = dataset.get_split()['valid']
	positive_samples = val_data[val_data['Y']==1]
	negative_samples = val_data[val_data['Y']==0]
	negative_samples = negative_samples.sample(positive_samples.shape[0])
	val_data = pd.concat([positive_samples.iloc[0:k,:],negative_samples.iloc[0:k,:]],axis=0)

	test_data = dataset.get_split()['test']

	train_data = build_corpus(train_data)
	val_data = build_corpus(val_data)
	test_data = build_corpus(test_data)

	
	return train_data, val_data, test_data
	


def make_regression_dataset(
	path:str = "../data/",
	name:str = "TAP",
	label_idx:int = 0
	):
	# assert label_idx in [0,1,2,3,4,"all"] "label index must be between 0 and 4"
	label_list = retrieve_label_name_list('TAP')
	dataset = Develop(name = 'TAP', label_name = label_list[label_idx])	
	data = dataset.get_data()
	ab = process_regression_antibody(data['Antibody'][0])
	

		


if __name__ == '__main__':
	make_classification_dataset()
	# make_regression_dataset()