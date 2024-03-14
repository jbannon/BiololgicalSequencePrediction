import sys
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from collections import Counter, OrderedDict
import data_utils as du
from functools import partial
from RNN_models import RecNN
import tqdm

def main():
	

	train, val, test = du.make_classification_dataset()
	
	vocab_map = du.make_vocab(train)
	
	collate_batch = partial(du.collate_batch, vocab_map = vocab_map)

	bs = 8
	train_dl = DataLoader(train, batch_size = bs, shuffle = True, collate_fn = collate_batch)
	val_dl = DataLoader(val, batch_size = bs, shuffle = False, collate_fn = collate_batch)
	test_dl = DataLoader(test, batch_size = bs, shuffle = False, collate_fn = collate_batch)
	

	vocab_size = len(vocab_map)
	embed_dim = 20
	rnn_hidden_size = 30
	fc_hidden_size = 30
	num_rec_layers = 2
	torch.manual_seed(1)
	model = RecNN(vocab_size, embed_dim, "gru",rnn_hidden_size,num_rec_layers,fc_hidden_size)
	
	# for seq_batch, label_batch, lengths in train_dl:
	# 	pred = model(seq_batch, lengths)
	# 	print(pred)
	# 	print(pred.shape)
	# 	print(pred[:,0])
	# 	sys.exit(1)
	loss_fn = nn.BCELoss()
	optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
	# optimizer = torch.optim.SGD(model.parameters(),lr = 0.01, momentum = 0.9)

	num_epochs = 1000

	def train(dataloader):
		model.train()
		total_acc, total_loss = 0,0

		for seq_batch, batch_labels, lengths in tqdm.tqdm(dataloader,leave=False):
			optimizer.zero_grad()
			pred = model(seq_batch, lengths)[:,0]
			# print(pred)
			# print(batch_labels)
			loss = loss_fn(pred, batch_labels)
			loss.backward()
			optimizer.step()
			total_acc += ( (pred>=0.5).float() == batch_labels).float().sum().item()
			total_loss += loss.item()*batch_labels.size(0)

		return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)

	def evaluate(dataloader):
		# print("\n--------")
		# print("eval")
		total_acc, total_loss = 0,0
		with torch.no_grad():
			for seq_batch, batch_labels, lengths in dataloader:
				pred = model(seq_batch,lengths)[:,0]
				# print(pred)
				# print(batch_labels)
				loss = loss_fn(pred, batch_labels)
				total_acc += ( (pred>=0.5).float() == batch_labels).float().sum().item()
			total_loss += loss.item()*batch_labels.size(0)
		return total_acc/len(dataloader.dataset), total_loss/len(dataloader.dataset)

	for epoch in range(num_epochs):
		
		acc_train, loss_train = train(train_dl)
		acc_val, loss_val = evaluate(val_dl)
		if acc_train == 1:
			sys.exit(1)
		print(f'Epoch {epoch} accuracy: {acc_train:.4f}'f' val_accuracy: {acc_val:.4f}')
		# print(f'Epoch {epoch} loss: {loss_train:.4f}'f' val_loss: {loss_val:.4f}')

	

if __name__ == '__main__':
	main()