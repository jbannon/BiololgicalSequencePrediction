import tqdm 
from torchtext.vocab import vocab
import torch
from torchtext.datasets import IMDB
import re
import torch.nn as nn
from collections import Counter, OrderedDict
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from RNN_models import RecNN

train_dataset = IMDB(split = "train")
test_dataset = IMDB(split = "test")





torch.manual_seed(1)

train_dataset, valid_dataset = random_split(list(train_dataset),[20000,5000])

def tokenizer(text):
	text = re.sub('<[^>]*>','',text)
	emoticons = re.findall(
		'(?::|;|=)(?:-)?(?:\)|\(|D|P)',text.lower())

	text = re.sub('[\W]+', ' ',text.lower()) + ' '.join(emoticons).replace('-','')
	tokenized = text.split()
	return tokenized

token_counts = Counter()

for label, line in train_dataset:
	tokens = tokenizer(line)
	# print(tokens)
	token_counts.update(tokens)

sort_by_freq = sorted(token_counts.items(), key = lambda x:x[1], reverse = True)
ordered_dict = OrderedDict(sort_by_freq)

vocab = vocab(ordered_dict)
vocab.insert_token("<pad>",0)
vocab.insert_token("<unk>",1)
vocab.set_default_index(1)

text_pipeline = lambda x: [vocab[token] for token in tokenizer(x)]
label_pipeline = lambda x: 1. if x == 'pos' else 0. 

def collate_batch(batch):
	label_list, text_list, lengths = [],[],[]
	for _label, _text in batch:
		
		label_list.append(label_pipeline(_label))
		
		processed_text = torch.tensor(text_pipeline(_text),dtype = torch.int64)
		
		text_list.append(processed_text)

		lengths.append(processed_text.size(0))

	label_list = torch.tensor(label_list)
	
	lengths = torch.tensor(lengths)

	padded_text_list = nn.utils.rnn.pad_sequence(
		text_list, batch_first = True)

	return padded_text_list, label_list, lengths

batch_size = 512

train_dl = DataLoader(train_dataset, batch_size= batch_size, 
	shuffle = True, collate_fn = collate_batch)

valid_dl = DataLoader(valid_dataset, batch_size = batch_size,
	shuffle = False, collate_fn = collate_batch)

test_dl = DataLoader(test_dataset, batch_size = batch_size,
	shuffle = False, collate_fn = collate_batch)


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

vocab_size = len(vocab)
embed_dim = 20 
rnn_hidden_size = 64 
fc_hidden_size = 64 
torch.manual_seed(1)
model = RecNN(vocab_size,embed_dim,'lstm',rnn_hidden_size,1,fc_hidden_size)
num_epochs = 10
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)

def train(dataloader):
	model.train()
	total_acc, total_loss = 0, 0
	for text_batch, label_batch, lengths in tqdm.tqdm(dataloader,leave=False):
		optimizer.zero_grad()
		pred = model(text_batch, lengths)[:,0]
		loss = loss_fn(pred, label_batch)
		loss.backward()
		optimizer.step()
		total_acc += ( (pred>=0.5).float() == label_batch).float().sum().item()
		total_loss += loss.item()*label_batch.size(0)
	return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)


def evaluate(dataloader):
		# print("\n--------")
		# print("eval")
		total_acc, total_loss = 0,0
		with torch.no_grad():
			for text_batch, label_batch, lengths in dataloader:
				pred = model(text_batch,lengths)[:,0]
				# print(pred)
				# print(batch_labels)
				loss = loss_fn(pred, label_batch)
				total_acc += ( (pred>=0.5).float() == label_batch).float().sum().item()
			total_loss += loss.item()*label_batch.size(0)
		return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)

for epoch in range(num_epochs):
		
		acc_train, loss_train = train(train_dl)
		acc_val, loss_val = evaluate(valid_dl)
		if acc_train == 1:
			sys.exit(1)
		print(f'Epoch {epoch} accuracy: {acc_train:.4f}'f' val_accuracy: {acc_val:.4f}')
	