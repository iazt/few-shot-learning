import torch
import numpy as np
import sys
from src.few_shot_episode import evalDataset
from torch.utils.data import Dataset, DataLoader

def euclidean_distance(x1, x2):
	return torch.sqrt(torch.sum((x1-x2)**2, dim=1))

class ned():
	def __init__(self, embedding_support, labels_support, k, n_classes):
		self.T = torch.var(embedding_support)
		self.embedding_support = embedding_support
		self.labels_support = labels_support
		self.k = k
		self.n_classes = n_classes
	def k_nearest(self, input_embedding):
		distances = euclidean_distance(self.embedding_support, input_embedding)
		ordered_idxs = torch.sort(distances, dim = 0)
		ordered_embedding = self.embedding_support[ordered_idxs.indices]
		ordered_labels = self.labels_support[ordered_idxs.indices]
		return ordered_embedding[:self.k-1], ordered_labels[:self.k-1]

	def calculate_probabilities(self, input_embedding):
		k_nearest_embedding, k_nearest_labels = self.k_nearest(input_embedding)
		probabilities = []		
		for cl in range(self.n_classes):
			term = torch.exp(-euclidean_distance(k_nearest_embedding, input_embedding)/self.T)
			prob = torch.sum(term[k_nearest_labels==cl]) / torch.sum(term)
			probabilities.append(prob)
		return probabilities


def eval_Ned(net, test_set, n_episodes = 600):
	classes  = [i for i in range(20)]
	k = 5
	N = 5
	idxs = [i for i in range(600)] 
	accuracy = []
	labels = torch.tensor(test_set.labels)
	images = test_set.images

	for i in range(n_episodes):
		choosen_classes = np.random.choice(classes, N, replace = False)
		choosen_idxs = np.random.choice(idxs, 21, replace = False)
		support_idxs = choosen_idxs[:k]  #indices de los ejemplos en el dataset de test
		support_idxs = np.array([support_idxs + clas*600 for clas in choosen_classes]).reshape(-1)
		query_idxs = choosen_idxs[k:]
		query_idxs = np.array([query_idxs+ clas*600 for clas in choosen_classes]).reshape(-1)


		support_set_im = images[support_idxs]
		support_set_labels = labels[support_idxs]
		query_set_im = images[query_idxs]
		query_set_labels = labels[query_idxs]

		support_dataset = evalDataset(support_set_im, support_set_labels)
		query_dataset = evalDataset(query_set_im, query_set_labels)
		support_loader = DataLoader(support_dataset, batch_size = 4, shuffle=True, num_workers=4, pin_memory=True)
		query_loader = DataLoader(query_dataset, batch_size = 16, shuffle=False, num_workers=4, pin_memory=True)
		total_query = len(query_loader)*query_loader.batch_size

		support_embeddings = []
		support_labels = []
		query_embeddings = []
		query_labels = []
		net.eval()
		for i, data in enumerate(support_loader, 0):
			with torch.no_grad():
				labe = data[0].cuda()
				inputs = data[1].float().cuda()
				feat = net.forward_embedding(inputs)
				support_embeddings.append(feat)
				support_labels.append(labe)

		for i, data in enumerate(query_loader, 0):
			with torch.no_grad():
				labe = data[0].cuda()
				inputs = data[1].float().cuda()
				feat = net.forward_embedding(inputs)
				query_embeddings.append(feat)
				query_labels.append(labe)

		ned_algorithm = ned(torch.cat(support_embeddings, dim = 0), torch.cat(support_labels, dim =0),5, 5) 
		acc = 0
		for embed, label in zip(torch.cat(query_embeddings,dim = 0), torch.cat(query_labels,dim=0)):
			prob =  ned_algorithm.calculate_probabilities(embed)
			prob = torch.stack(prob)
			pred = torch.argmax(prob)
			acc += pred == label
		accuracy.append(acc/total_query)
		info = f'Test Acc:{acc/total_query*100:02.1f}%\n'
		sys.stdout.write(info)
	return accuracy