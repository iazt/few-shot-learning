import torch

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

