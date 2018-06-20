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

def to_var(indexes):
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([256, 256], Image.LANCZOS)
    image = image.convert('RGB')
    if transform is not None:
        image = transform(image).unsqueeze(0)    
    return image

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


def evaluateRandomly(encoder, decoder, out_vocab, inp_vocab, n=10):
    testing_reps = os.listdir(args.testing_rep_dir)
    testing_reps = sorted(testing_reps ,key = numericalSort)

    for i,input_file in enumerate(testing_reps):
        # input_file = reps[i]
        with open(args.testing_rep_dir + '/' + input_file,"rb") as f:
            current_pos,state,action,new_pos,lives = pickle.load(f)
        data = (current_pos,state,action,new_pos,lives)

        # with open("./png/Test_Rep.txt") as f: 
        #         content = f.readlines()
        # state = []
        # for k,line in enumerate(content):
        #     nums = line.split()
        #     for i,num in enumerate(nums):
        #         nums[i] = str(num)
        #     if k==0:
        #         current_pos = nums
        #     elif k==15:
        #         action = nums[0]
        #     elif k==16: 
        #         new_pos = nums
        #     elif k==17:
        #         lives = 'T'
        #     else:
        #         state.append(nums)
        # state=np.array([np.array(xi) for xi in state])
        # data = (current_pos,state,action,new_pos,lives)
        # print(data)
        # exit(0)
        final_rep = []
        for i,r in enumerate(data):
            if i == 1:
                r = list(r.flatten())
                final_rep.extend(r)
            elif i==2:
                final_rep.append(r)
            elif i==4:
                final_rep.append(r)
            else:
                final_rep.extend(r)
        print(final_rep)
        final_rep = np.array(final_rep)
        feature = []
        for r in final_rep:
            feature.append(inp_vocab(r))
        feature = to_var(feature)
        print('>', data)
        output_words, _ = evaluate(encoder, decoder, feature, out_vocab)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
        exit(0)
def evaluate(encoder, decoder, input_tensor, out_vocab, max_length=244, max_decode = 300):
    with torch.no_grad():
        # input_tensor = tensorFromSentence(input_lang, sentence)
        # input_length = input_tensor.size()[0]

        # input_tensor = to_var(sentence)
        input_length = input_tensor.size()[0]
        # print(input_length)
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



        beam_size = 10

        candidates = []
        all_candidates = []
        for di in range(max_decode):
            if di==0:    
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                # decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(beam_size)
                for k in range(beam_size):
                    candidates.append([topi[0][k].unsqueeze(0), topv[0][k].cpu().data.numpy() , decoder_hidden])
            else:
                all_candidates = []
                break_flag = True
                for candidate in candidates:
                    if candidate[0][len(candidate[0])-1] != EOS_token:
                        break_flag = False
                if break_flag == True:
                    # print("inside")
                    break
                for k in range(beam_size):
                    candidate = candidates[k]
                    if candidate[0][len(candidate[0])-1] == EOS_token:
                        all_candidates.append(candidate)
                    else:
                        # print(candidate)
                        decoder_input = candidate[0][len(candidate[0])-1]
                        decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, candidate[2], encoder_outputs)
                        # decoder_attentions[di] = decoder_attention.data
                        topv, topi = decoder_output.data.topk(beam_size)

                        for j in range(beam_size):
                            new_candidate = [torch.cat((candidate[0],topi[0][j].unsqueeze(0)),0),(candidate[1] + topv[0][j].cpu().data.numpy())/(len(candidate[0]) + 1), decoder_hidden]
                            all_candidates.append(new_candidate)
                ordered = sorted(all_candidates, key=lambda tup:tup[1], reverse = True)
                candidates = ordered[:beam_size]
        beam = []
        for i,candidate in enumerate(candidates):
            # print(candidate)
            sampled_ids = candidate[0]
            # print(sampled_ids)
            print("Beam # " + str(i))
            beam = []
            for id in sampled_ids:
                if id == EOS_token:
                    beam.append('<end>')
                    break
                else: 
                    beam.append(out_vocab.idx2word[id.item()])
            output_sentence = ' '.join(beam)
            print(output_sentence)

        sampled_ids = candidates[0][0]
        for id in sampled_ids:
            # print(id)
            if id == EOS_token:
                decoded_words.append('<end>')
                break
            else: 
                decoded_words.append(out_vocab.idx2word[id.item()])

    return decoded_words, decoder_attentions[:di + 1]
def main(args):
    # Image preprocessing
    # transform = transforms.Compose([
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(), 
    #     transforms.Normalize((0.485, 0.456, 0.406), 
    #                          (0.229, 0.224, 0.225))])

    transform = transforms.Compose([
        # transforms.ColorJitter(contrast = 0.3,saturation = 0.3),
        # transforms.RandomChoice([transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()]),
        transforms.RandomAffine(0,translate = (0.1,0.1)),
        transforms.ToTensor(), 
        # transforms.Normalize((0.8, 0.7, 0.8), 
        #                     (1, 1, 1))
        ])
    
    # Load vocabulary wrapper
    with open(args.inp_vocab_path, 'rb') as f:
        inp_vocab = pickle.load(f)
    with open(args.out_vocab_path, 'rb') as f:
        out_vocab = pickle.load(f)

    # Build Models
    # print(len(vocab))
    # encoder = EncoderCNN(args.embed_size)
    # encoder.eval()  # evaluation mode (BN uses moving mean/variance)
    # encoder = get_deform_cnn(True)
    # encoder = encoder.cuda()

    encoder = EncoderRNN(len(inp_vocab), args.hidden_size,device).to(device)
    decoder = AttnDecoderRNN(args.hidden_size, len(out_vocab),device, dropout_p=0.1).to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))

    evaluateRandomly(encoder,decoder,out_vocab,inp_vocab)
    # decoder = DecoderRNN(args.embed_size, args.hidden_size, 
    #                      len(vocab), args.num_layers)
    

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=False,
                        help='input current image for generating caption')
    # parser.add_argument('--next_image', type=str, required=True,
    #                     help='input next image for generating caption')
    parser.add_argument('--encoder_path', type=str, default='./models/seq2seq_encoder_v2-91-500.pkl',
                        help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='./models/seq2seq_decoder_v2-91-500.pkl',
                        help='path for trained decoder')
    parser.add_argument('--out_vocab_path', type=str, default='./data/vocab_frogger.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--inp_vocab_path', type=str, default='./data/input_vocab.pkl',
                    help='path for vocabulary wrapper')
    parser.add_argument('--testing_rep_dir', type=str, default='./data/FroggerSymbolicRepresentationTesting' ,
                        help='directory for resized images')
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=244,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    args = parser.parse_args()
    main(args)