import sys
import os
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# data_list = [Data(...), ..., Data(...)]
# loader = DataLoader(data_list, batch_size=32)
from torch_geometric.datasets import TUDataset
from torch_geometric.data import InMemoryDataset, download_url
import os.path as osp
from typing import Callable, List, Optional
import torch
from torch_geometric.data import download_url
import pandas as pd


def fetch_york_networkx(
	root:str
	)->None:
	pass

def fetch_york_pyg(
	root:str) -> None:
	pass

residue_map = {'U':1.0, 'A':2.0,'C':3.0,'G':4.0,'O':5.0}
if __name__ == '__main__':
	zipname = "York RNA Graph Dataset.zip".replace(" ","%20")
	url = "https://www.cs.york.ac.uk/cvpr/datasets/{zn}".format(zn=zipname)
	raw_dir = "../data/york/raw"

	processed_dir = "../data/york/processed"
	os.makedirs(raw_dir,exist_ok = True)
	os.makedirs(processed_dir,exist_ok = True)
	download_url(url,raw_dir)

	# os.system("unzip -qq {rf}/{z} -d {rf}/york_data".format(rf = raw_dir, z = zipname, p = processed_dir))
	# os.system("rm {rf}/{z}".format(rf=raw_dir,z=zipname))

	with open("{rf}/york_data/classes.txt".format(rf=raw_dir),"r") as istream:
		lines = istream.readlines()
		class_labels = [int(x.rstrip()) for x in lines]

	rna_files = [x for x in os.listdir("{rf}/york_data/rna3178/".format(rf=raw_dir))]
	data_list = []
	print(pd.value_counts(class_labels))
	sys.exit(1)
	for i in range(1,len(class_labels)+1):
		

		label = class_labels[i-1]-1
		if label == -1:
			continue
		

		rna_file = "rna{n}.txt".format(n=i)
		
		with open("{rf}/york_data/rna3178/{f}".format(rf=raw_dir,f=rna_file)) as istream:
			lines = istream.readlines()
			lines = [x.rstrip() for x in lines]
		
		num_verts = int(lines[0])

		vertex_features = torch.empty((num_verts,7),dtype=torch.float)

		for v in range(1, num_verts+1):
			features = lines[v].split(" ")
			features[0] = residue_map[features[0]]
			vertex_features[v-1] = torch.tensor([float(x) for x in features],dtype=torch.float)
		

		edge_line= num_verts+1
		num_edges = lines[edge_line]

		edge_list = []
		for e in range(edge_line+1,len(lines)):
			endpoints = [int(x)-1 for x in lines[e].split(" ")]

			edge_list.append([x for x in endpoints])
			endpoints.reverse()
			edge_list.append([x for x in endpoints])
		edge_index = torch.tensor(edge_list,dtype=torch.long)
		
		data = Data(x=vertex_features, edge_index=edge_index.t().contiguous(), y = torch.tensor(label))
		data_list.append(data)
	print(data_list)
	torch.save(data_list, processed_dir+"york.pt")
