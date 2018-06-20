import torch.utils.data as data
import torch
import torchvision.transforms as transforms
import torchvision
import nltk
from PIL import Image
import numpy as np
from build_vocab_v2 import Vocabulary
class combined_dataset(data.Dataset):
	def __init__(self,sasr_dataset,vocab,input_vocab,transform,reverse):
		self.sasr_dataset = sasr_dataset
		self.vocab = vocab
		self.input_vocab = input_vocab
		self.transform = transform
		self.reverse = reverse
	def __getitem__(self,index):
		caption = []
		caption.append(self.vocab('<start>'))
		tokens = nltk.tokenize.word_tokenize(str(self.sasr_dataset[index].rationalization).lower())
		caption.extend([self.vocab(token) for token in tokens])
		caption.append(self.vocab('<end>'))
		# print(tokens)
		target = torch.Tensor(caption)
		# image = self.sasr_dataset[index].subtracted_images
		# image = image.convert('RGB')
		# print(index)
		# print(self.sasr_dataset[index].symbolic_rep)
		rep = self.sasr_dataset[index].symbolic_rep
		# print(rep)
		final_rep = []
		for i,r in enumerate(rep[0]):
			if i == 1:
				r = list(r.flatten())
				final_rep.extend(r)
			elif i==2:
				final_rep.append(r)
			elif i==4:
				final_rep.append(r)
			else:
				final_rep.extend(r)
		# final_rep = np.array(final_rep)
		# print(final_rep)
		# # exit(0)
		# print(final_rep[1])
		# print(self.input_vocab.word2idx)
		# print(self.input_vocab(final_rep[1]))
		# print(len(final_rep))
		# exit(0)
		for i,ch in enumerate(final_rep):
			# print(final_rep[i])
			final_rep[i] = self.input_vocab(final_rep[i])
		if self.reverse:
			final_rep = list(reversed(final_rep))
		final_rep = np.array(final_rep)
		# print(final_rep)
		image = torch.Tensor(final_rep)
		# print(image)
		# exit(0)
		# image.save('zzzz.png')
		# exit(0)
		# if self.transform is not None:
		# 	image = self.transform(image) 

		return image,target
	def __len__(self):
		return len(self.sasr_dataset)