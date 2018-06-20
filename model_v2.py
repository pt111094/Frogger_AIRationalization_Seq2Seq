import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import torch.nn.functional as F
from Attention import Attn


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()
        
    def init_weights(self):
        """Initialize the weights."""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
        
    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)
        # print(features)
        # exit(0)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features
    
    
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        return outputs

    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(30):                                      # maximum sampling length
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                         # (batch_size, 1, embed_size)
        # print(sampled_ids)
        sampled_ids = torch.cat(sampled_ids, 0)                  # (batch_size, 20)
        # print(sampled_ids)
        # exit(0)
        return sampled_ids.squeeze()

class AttnDecoderRNN(nn.Module):
    def __init__(self,  feature_size, hidden_size, vocab_size, num_layers):
        super(AttnDecoderRNN, self).__init__()
        #Define parameters

        #Define layers
        self.embed = nn.Embedding(vocab_size, feature_size)
        self.init_layer = nn.Linear(feature_size, hidden_size)
        self.attn = Attn('general', feature_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.ctx2out = nn.Linear(feature_size, feature_size)
        self.h2out = nn.Linear(hidden_size, feature_size)
        self.out = nn.Linear(feature_size, vocab_size)
    
    def decode_lstm(self, input_word, context, hidden, lstm_out):

        hidden = hidden.squeeze(0)
        out = self.h2out(hidden)
        context = context.squeeze(1)
        out += self.ctx2out(context)
        out += input_word.squeeze(1)
        out = F.tanh(out)
        out = self.out(out)
       
        return out

    def init_lstm(self, features):
        print(features)
        exit(0)
        sums = torch.sum(features, 2)
        out = torch.mul(sums, 1/features.size(2))
        out = out.unsqueeze(0) # 1, batch, feature_size
        out = self.init_layer(out.squeeze(0)).unsqueeze(0)
        out = F.tanh(out)
        return out, out 


    def forward(self, features, captions, lengths):
        # print(features)
        max_length = max(lengths)
        embed = self.embed(captions)
        h, c= self.init_lstm(features)
        arr = [] 

        for i in range(max_length):            
            if i == 0 :
                input_word = Variable(torch.zeros(embed.size(0), embed.size(2)).cuda())
            else: 
                input_word = embed[:,i-1]
            context = self.attn(h, features)           
            lstm_input = torch.cat((context, input_word.unsqueeze(1)),2)
            lstm_out, (h,c) = self.lstm(lstm_input, (h,c))
            out = self.decode_lstm(input_word,context, h, lstm_out).unsqueeze(1)
            arr += [out]

        return torch.cat(arr,1)

    def sample(self, features, embed_size):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        h,c = self.init_lstm(features)

        for i in range(150):                                      # maximum sampling length
            if i == 0:
                x = Variable(torch.rand(1,1,embed_size)).cuda(0)
            else: 
                x = self.embed((predicted)).unsqueeze(1)
            context = self.attn(h, features)
            lstm_input = torch.cat((context, x) ,2)
            lstm_out, (h,c) = self.lstm(lstm_input, (h,c))          # (batch_size, 1, hidden_size), 
            out = self.decode_lstm(x, context, h, lstm_out)
            predicted = out.max(1)[1]
            sampled_ids.append(predicted.cpu().data.numpy()[0])
        #sampled_ids = torch.cat(sampled_ids, 1)                  # (batch_size, 20)
        return sampled_ids
    def sample_beam_search(self, features, embed_size):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(0).unsqueeze(0)
        beam_size = 5
        candidates = []
        all_candidates = []
        h,c = self.init_lstm(features)
        for i in range(30):                                      # maximum sampling length
            if i==0:
                x = Variable(torch.rand(1,1,embed_size)).cuda(0)
                context = self.attn(h, features)
                lstm_input = torch.cat((context, x) ,2)
                lstm_out, (h,c) = self.lstm(lstm_input, (h,c))          # (batch_size, 1, hidden_size), 
                outputs = self.decode_lstm(x, context, h, lstm_out)
                predictions = torch.topk(outputs,beam_size)
                for k in range(beam_size):
                    candidates.append([predictions[1][0][k], predictions[0][0][k].cpu().data.numpy()[0] , h , c])   
            else: 
                all_candidates = []
                for k in range(beam_size):
                    candidate = candidates[k]
                    x = self.embed((candidate[0][len(candidate[0])-1])).unsqueeze(1)
                    context = self.attn(candidate[2], features)
                    lstm_input = torch.cat((context, x) ,2)
                    lstm_out, (h,c) = self.lstm(lstm_input, (candidate[2],candidate[3]))
                    outputs = self.decode_lstm(x, context, h, lstm_out)
                    predictions = torch.topk(outputs,beam_size)
                    for k in range(beam_size):
                        new_candidate = [torch.cat((candidate[0],predictions[1][0][k]),0),candidate[1] + predictions[0][0][k].cpu().data.numpy()[0], h, c]
                        all_candidates.append(new_candidate)
                ordered = sorted(all_candidates, key=lambda tup:tup[1], reverse = True)
                candidates = ordered[:beam_size]
        sampled_ids = candidates[0][0].cpu().data.numpy()
        # print(sampled_ids)
        return sampled_ids.squeeze()

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5,
                     stride=stride, padding=2, bias=False)

def convLayer(in_channels, out_channels, stride=1, kernel_size = 5):
    return nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size,
                     stride=stride, padding=(kernel_size-1)/2, bias=False)
# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = convLayer(in_channels, out_channels, 1 , 3)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = convLayer(out_channels, out_channels, stride, 3)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # print(residual)
        if self.downsample:
            # print("inside")
            residual = self.downsample(x)
        # print(out)
        # print(residual)
        out += residual
        # print(out.shape)
        out = self.relu(out)
        return out

# ResNet Module for Show Attend and Tell model 
class AttnEncoder(nn.Module):
    def __init__(self, block, layers, embed_size):
        super(AttnEncoder, self).__init__()
        self.in_channels = 32
        self.conv = convLayer(3, 32, 1, 3)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 32, layers[0])
        self.layer2 = self.make_layer(block, 64, layers[0],3)
        self.layer3 = self.make_layer(block, embed_size, layers[1],3)
        #self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        self.fc.weight.data.normal_(0.0, 0.02)
        self.fc.bias.data.fill_(0)
        self.fc2.weight.data.normal_(0.0, 0.02)
        self.fc2.bias.data.fill_(0)

    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                convLayer(self.in_channels, out_channels, stride=stride, kernel_size = 3),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        # print("after1")
        out = self.layer2(out)
        # print("after2")
        out = self.layer3(out)
        # print(out.shape)
        out = out.view(out.size(0), out.size(1), -1)
        # print(out.shape)
        return out
