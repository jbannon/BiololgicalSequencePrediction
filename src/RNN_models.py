import torch.nn as nn
import torch

class RecNN(nn.Module):
	"""
	Generic Recurrent Neural Net Class
		-> supports parameterizing the kind of layer
		-> 

	"""

	def __init__(self,
		vocab_size:int,
		embedding_dim:int,
		net_type:str,
		rnn_hidden_dim:int,
		num_rec_layers:int,
		fc_hidden_dim:int,
		pad_idx:int = 0,
		bidirection:bool = False
		)-> None:

		super().__init__()
		net_type = net_type.lower()
		assert net_type in ['rnn','lstm','gru'], "net type must be one of: ['rnn','lstm','gru']"

		self.net_type = net_type
		self.vocab_size = vocab_size
		self.embedding_dim = embedding_dim
		self.rnn_hidden_dim = rnn_hidden_dim
		self.fc_hidden_dim = fc_hidden_dim
		
		self.pad_idx = pad_idx
 
		self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_idx)
		self.build_recurrent_layer()
		self.fc1 = nn.Linear(rnn_hidden_dim,fc_hidden_dim)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(fc_hidden_dim,1)
		self.sigmoid = nn.Sigmoid()

	def embed_sequence(self,x):
		return self.embedding(x)
		


	def build_recurrent_layer(self):
		if self.net_type == 'rnn':
			self.rec_layer= nn.RNN(self.embedding_dim,self.rnn_hidden_dim, batch_first = True)
		elif self.net_type == 'lstm':
			self.rec_layer= nn.LSTM(self.embedding_dim,self.rnn_hidden_dim, batch_first = True)
		else:
			self.rec_layer= nn.GRU(self.embedding_dim,self.rnn_hidden_dim, batch_first = True)


	def forward(self, text, lengths):
		out = self.embedding(text)
		out = nn.utils.rnn.pack_padded_sequence(out, 
			lengths.cpu().numpy(), enforce_sorted = False, batch_first = True)

		if self.net_type == 'lstm':
			out, (hidden, cell) = self.rec_layer(out)
		else:
			out, hidden = self.rec_layer(out)
		out = hidden[-1,:,:]
		out = self.fc1(out)
		# out = self.relu(out)
		# out = self.fc2(out)
		out = self.sigmoid(out)
		return out