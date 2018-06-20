import argparse
import torch
import torch.nn as nn
import re
import numpy as np
import time
import math
import random
import os
import pickle
# from data_loader import get_loader
# from data_loader import get_images
# from sasr_data_loader import data_loader
# from sasr_data_loader import load_data
from sasr_data_loader_v4 import SASR_Data_Loader
from img_to_vec import Img2Vec
from build_vocab_v2 import Vocabulary
from model_v5_seq2seq import EncoderRNN, AttnDecoderRNN
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from torch import optim

from layers import ConvOffset2D
from model_v4 import get_cnn, get_deform_cnn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 1
EOS_token = 2
teacher_forcing_ratio = 0.5
import re
numbers = re.compile(r'(\d+)')

def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

def to_var(indexes):
	return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def evaluateRandomly(encoder, decoder, out_vocab, inp_vocab, n=10, reverse = False):
	testing_reps = os.listdir(args.testing_rep_dir)
	testing_reps = sorted(testing_reps ,key = numericalSort)

	reps = random.sample(testing_reps,n)
	for i in range(n):
		input_file = reps[i]
		with open(args.testing_rep_dir + '/' + input_file,"rb") as f:
			current_pos,state,action,new_pos = pickle.load(f)
		# print(state)
		# print(action)
		data = (current_pos,state,action,new_pos)
		final_rep = []
		for i,r in enumerate(data):
			# print(r)
			# print(rep[0][0])
			# print("*********************" + str(i))
			if i == 1:
				r = list(r.flatten())
				final_rep.extend(r)
			elif i==2:
				final_rep.append(r)
			else:
				final_rep.extend(r)
		final_rep = np.array(final_rep)
		feature = []
		for r in final_rep:
			# print(inp_vocab(r))
			# print(type(inp_vocab(r)))
			feature.append(inp_vocab(r))
		if reverse:
			feature = list(reversed(feature))
		# print(type(feature[0]))
		# image = torch.Tensor(final_rep)
		feature = to_var(feature)
		print('>', data)
		# print(len(final_rep))
		# print('=', pair[1])
		output_words, attentions = evaluate(encoder, decoder, feature, out_vocab)
		output_sentence = ' '.join(output_words)
		print('<', output_sentence)
		print('')


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=244):
	encoder_hidden = encoder.initHidden()

	encoder_optimizer.zero_grad()
	decoder_optimizer.zero_grad()

	input_length = input_tensor.size(0)
	target_length = target_tensor.size(0)

	encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

	loss = 0
	# print(input_length)
	for ei in range(input_length):
		# print(input_tensor[ei])
		encoder_output, encoder_hidden = encoder(
		    input_tensor[ei], encoder_hidden)
		# exit(0)
		encoder_outputs[ei] = encoder_output[0, 0]

	decoder_input = torch.tensor([[SOS_token]], device=device)

	decoder_hidden = encoder_hidden

	use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

	if use_teacher_forcing:
	    # Teacher forcing: Feed the target as the next input
	    for di in range(target_length):
	        decoder_output, decoder_hidden, decoder_attention = decoder(
	            decoder_input, decoder_hidden, encoder_outputs)
	        loss += criterion(decoder_output, target_tensor[di])
	        decoder_input = target_tensor[di]  # Teacher forcing

	else:
	    # Without teacher forcing: use its own predictions as the next input
	    for di in range(target_length):
	        decoder_output, decoder_hidden, decoder_attention = decoder(
	            decoder_input, decoder_hidden, encoder_outputs)
	        topv, topi = decoder_output.topk(1)
	        decoder_input = topi.squeeze().detach()  # detach from history as input

	        loss += criterion(decoder_output, target_tensor[di])
	        if decoder_input.item() == EOS_token:
	            break

	loss.backward()

	encoder_optimizer.step()
	decoder_optimizer.step()

	return loss.item() / target_length

def evaluate(encoder, decoder, input_tensor, out_vocab, max_length=244):
	with torch.no_grad():
		# input_tensor = tensorFromSentence(input_lang, sentence)
		# input_length = input_tensor.size()[0]

		# input_tensor = to_var(sentence)
		input_length = input_tensor.size()[0]
		print(input_length)
		encoder_hidden = encoder.initHidden()

		encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

		for ei in range(input_length):
			encoder_output, encoder_hidden = encoder(input_tensor[ei],
			                                         encoder_hidden)
			encoder_outputs[ei] += encoder_output[0, 0]

		decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

		decoder_hidden = encoder_hidden

		decoded_words = []
		decoder_attentions = torch.zeros(max_length, max_length)

		for di in range(max_length):
			decoder_output, decoder_hidden, decoder_attention = decoder(
			    decoder_input, decoder_hidden, encoder_outputs)
			decoder_attentions[di] = decoder_attention.data
			topv, topi = decoder_output.data.topk(1)
			if topi.item() == EOS_token:
				decoded_words.append('<end>')
				break
			else:
				decoded_words.append(out_vocab.idx2word[topi.item()])

			decoder_input = topi.squeeze().detach()

	return decoded_words, decoder_attentions[:di + 1]


def timeSince(since, percent):
	now = time.time()
	s = now - since
	es = s / (percent)
	rs = es - s
	return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def main(args):
	if not os.path.exists(args.model_path):
	    os.makedirs(args.model_path)
	with open(args.out_vocab_path, 'rb') as f:
	    out_vocab = pickle.load(f)
	with open(args.input_vocab_path, 'rb') as f:
	    inp_vocab = pickle.load(f)
	# data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
	#                          transform, args.batch_size,
	#                          shuffle=True, num_workers=args.num_workers) 
	transform = transforms.Compose([
	    # transforms.ColorJitter(contrast = 0.3,saturation = 0.3),
	    # transforms.RandomChoice([transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()]),
	    transforms.RandomAffine(0,translate = (0.1,0.1)),
	    transforms.ToTensor(), 
	    # transforms.Normalize((0.8, 0.7, 0.8), 
	    #                     (1, 1, 1))
    ])
	sasr_data_loader = SASR_Data_Loader(out_vocab,inp_vocab,transform)
	sasr_data_loader.load_data(args.data_file,args.init_flag)
	frogger_data_loader = sasr_data_loader.data_loader(args.batch_size, 
	                         transform,
	                         shuffle=True, reverse = args.reverse,num_workers=args.num_workers) 
	encoder = EncoderRNN(len(inp_vocab), args.hidden_size,device).to(device)
	decoder = AttnDecoderRNN(args.hidden_size, len(out_vocab),device, dropout_p=0.1).to(device)

	start = time.time()
	plot_losses = []
	print_loss_total = 0  # Reset every print_every
	plot_loss_total = 0  # Reset every plot_every
	encoder_optimizer = optim.SGD(encoder.parameters(), lr=args.learning_rate)
	decoder_optimizer = optim.SGD(decoder.parameters(), lr=args.learning_rate)

	criterion = nn.NLLLoss()
	total_step = len(frogger_data_loader)

	for epoch in range(args.num_epochs):
		for i,(images,captions,lengths) in enumerate(frogger_data_loader):
			###CONVERT IMAGES AND CAPTIONS INTO DEVICE TENSORS
			input_tensor = to_var(images)
			target_tensor = to_var(captions)

			loss = train(input_tensor, target_tensor, encoder,
		             decoder, encoder_optimizer, decoder_optimizer, criterion)
			print_loss_total += loss
			plot_loss_total += loss
			# print(loss)
			# if epoch%5=1: 
			if i % args.save_step == args.save_step-1:
				print_loss_avg = print_loss_total / args.save_step
				print_loss_total = 0
				# print(i)
				# print(args.num_epochs)
				# print(float(i) / float(args.num_epochs))
				print('Epoch [%d/%d], Step [%d/%d], Average loss: %.4f' % (epoch, args.num_epochs, i, total_step,print_loss_avg))

			# if i % plot_every == 0:
			#     plot_loss_avg = plot_loss_total / plot_every
			#     plot_losses.append(plot_loss_avg)
			# 	plot_loss_total = 0
			if epoch%10==0:
				if (i+1) % args.save_step == 0:
				    torch.save(decoder.state_dict(), 
				               os.path.join(args.model_path, 
				                            'seq2seq_decoder_v2-%d-%d.pkl' %(epoch+1, i+1)))
				    torch.save(encoder.state_dict(), 
				               os.path.join(args.model_path, 
				                        'seq2seq_encoder_v2-%d-%d.pkl' %(epoch+1, i+1))) 
	evaluateRandomly(encoder,decoder,out_vocab,inp_vocab)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/' ,
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 ,
                        help='size for randomly cropping images')
    parser.add_argument('--out_vocab_path', type=str, default='./data/vocab_frogger.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--input_vocab_path', type=str, default='./data/input_vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='./data/resized2014' ,
                        help='directory for resized images')
    parser.add_argument('--testing_rep_dir', type=str, default='./data/FroggerSymbolicRepresentationTesting' ,
                        help='directory for resized images')
    parser.add_argument('--data_file', type=str, default='Turk_Master_File.xlsx',
                        help='name of the excel file')
    parser.add_argument('--caption_path', type=str,
                        default='./data/annotations/captions_train2014.json',
                        help='path for train annotation json file')
    parser.add_argument('--log_step', type=int , default=10,
                        help='step size for prining log info')
    parser.add_argument('--save_step', type=int , default=500,
                        help='step size for saving trained models')
    
    # Model parameters
    # parser.add_argument('--embed_size', type=int , default=256 ,
    #                     help='dimension of word embedding vectors')
    parser.add_argument('--reverse',type = bool, default=False, 
    					help='Boolean that dictates whether or not the input is reversed')
    parser.add_argument('--embed_size', type=int , default=244 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    parser.add_argument('--init_flag', type=bool , default=False ,
                        help='Whether or not data has been initialized')
    
    
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--rnn', default='LSTM', choices=['LSTM', 'GRU'], help='rnn module type')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)