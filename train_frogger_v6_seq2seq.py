###### Use default Resnet as a feature generator without any backprop

import argparse
import torch
import torch.nn as nn
import re
import numpy as np
import os
import pickle
import time
import math
import random
# from data_loader import get_loader
# from data_loader import get_images
# from sasr_data_loader import data_loader
# from sasr_data_loader import load_data
from sasr_data_loader_v4 import SASR_Data_Loader
from img_to_vec import Img2Vec
from build_vocab import Vocabulary
from model_v5_seq2seq import EncoderRNN, AttnDecoderRNN
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms
from torch import optim

from layers import ConvOffset2D
from model_v4 import get_cnn, get_deform_cnn

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
    
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_LENGTH = 100
def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
from sasr_data_loader_v4 import action_to_num

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([256, 256], Image.LANCZOS)
    image = image.convert('RGB')
    if transform is not None:
        image = transform(image).unsqueeze(0)  

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    decoder_input = torch.tensor([[SOS_token]])

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


def main(args):
    # Create model directory
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    transform = transforms.Compose([
        # transforms.ColorJitter(contrast = 0.3,saturation = 0.3),
        # transforms.RandomChoice([transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip()]),
        transforms.RandomAffine(0,translate = (0.1,0.1)),
        transforms.ToTensor(), 
        # transforms.Normalize((0.8, 0.7, 0.8), 
        #                     (1, 1, 1))
        ])
    
    # Load vocabulary wrapper.
    with open(args.out_vocab_path, 'rb') as f:
        out_vocab = pickle.load(f)
    with open(args.input_vocab_path, 'rb') as f:
        inp_vocab = pickle.load(f)
    # data_loader = get_loader(args.image_dir, args.caption_path, vocab, 
    #                          transform, args.batch_size,
    #                          shuffle=True, num_workers=args.num_workers) 
    sasr_data_loader = SASR_Data_Loader(out_vocab,inp_vocab,transform)
    sasr_data_loader.load_data(args.data_file,args.init_flag)
    frogger_data_loader = sasr_data_loader.data_loader(args.batch_size, 
                             transform,
                             shuffle=True, num_workers=args.num_workers, reversed = False) 
    # Build the models
    # encoder = EncoderCNN(args.embed_size)
    # encoder = get_deform_cnn(True)
    # encoder = encoder.cuda()
#     decoder = DecoderRNN(args.embed_size, args.hidden_size, 
#                          len(vocab), args.num_layers)

#     decoder = RNN(vocab_size=len(vocab), embed_size=args.embed_size, num_output=args.classes, rnn_model=args.rnn,
#             use_last=( not args.mean_seq),
# hidden_size=args.hidden_size, embedding_tensor=None, num_layers=args.num_layers, batch_first=True)
    encoder = EncoderRNN(len(inp_vocab), args.hidden_size)
    decoder = AttnDecoderRNN(args.hidden_size, len(out_vocab), dropout_p=0.1)
    if torch.cuda.is_available():
        encoder.cuda()
        decoder.cuda()

    # Loss and Optimizer
    # criterion = nn.CrossEntropyLoss()
    criterion = nn.NLLLoss()
    params = list(decoder.parameters())
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)
    stransform = transforms.ToPILImage()

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=args.learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=args.learning_rate)

    img2vec = Img2Vec()
    total_step = len(frogger_data_loader)
    for epoch in range(args.num_epochs):
        for i,(images,captions,lengths) in enumerate(frogger_data_loader):
            # print(images)
            # exit(0)
            # images = to_var(images)

            # image1 = images.squeeze()
            # # print(image1)
            # c = stransform(image1)
            # features = img2vec.get_vec(c,True)
            # features = to_var(features)
            # images = images.to(device)
            # if (list(images.size())[0]!=1):
            # captions = to_var(captions)
            images = images.type(torch.LongTensor)
            images = to_var(images)
            captions = to_var(captions)
            loss = train(images,captions,encoder,decoder,encoder_optimizer,decoder_optimizer,criterion)
            print_loss_total += loss
            plot_loss_total += loss
            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                             iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0



            # decoder.zero_grad()
            # # encoder.zero_grad()
            # # features = encoder(images)
            # features = images
            # outputs = decoder(features, captions, lengths)
            # captions = captions.view(-1)
            # outputs = outputs.view(-1,len(vocab))
            # loss = criterion(outputs, captions)
            # loss.backward()
            # optimizer.step()

            # # Print log info
            # if i % args.log_step == 0:
            #     print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f, Perplexity: %5.4f'
            #           %(epoch, args.num_epochs, i, total_step, 
            #             loss.data[0], np.exp(loss.data[0]))) 
                
            # # Save the models
            # if (i+1) % args.save_step == 0:
            #     torch.save(decoder.state_dict(), 
            #                os.path.join(args.model_path, 
            #                             'decoder-%d-%d.pkl' %(epoch+1, i+1)))
                # torch.save(encoder.state_dict(), 
                #            os.path.join(args.model_path, 
                #                     'encoder-%d-%d.pkl' %(epoch+1, i+1)))                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models_reversed/' ,
                        help='path for saving trained models')
    parser.add_argument('--crop_size', type=int, default=224 ,
                        help='size for randomly cropping images')
    parser.add_argument('--out_vocab_path', type=str, default='./data/vocab_frogger.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--input_vocab_path', type=str, default='./data/input_vocab.pkl',
                        help='path for vocabulary wrapper')
    parser.add_argument('--image_dir', type=str, default='./data/resized2014' ,
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
    parser.add_argument('--embed_size', type=int , default=243 ,
                        help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512 ,
                        help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 ,
                        help='number of layers in lstm')
    parser.add_argument('--init_flag', type=bool , default=False ,
                        help='Whether or not data has been initialized')
    
    
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--rnn', default='LSTM', choices=['LSTM', 'GRU'], help='rnn module type')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    args = parser.parse_args()
    print(args)
    main(args)