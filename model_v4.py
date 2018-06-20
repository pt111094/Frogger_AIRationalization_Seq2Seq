from __future__ import absolute_import, division
import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable



import torch.nn.functional as F
from layers import ConvOffset2D

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.pooling = nn.MaxPool2d(2,stride = 2)
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
        features = Variable(features.data)
        features = self.pooling(features)
        # print(features)
        features = features.view(features.size(0),-1)
        # print(features)
        # print(resnet.fc.in_features)
        features = self.bn(self.linear(features))
        return features
        # with torch.no_grad():
        # features = self.resnet(images)
        # features = features.reshape(features.size(0), -1)
        # features = self.bn(self.linear(features))
        # return features
    

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # conv11
        self.conv11 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(32)

        # conv12
        self.conv12 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.bn12 = nn.BatchNorm2d(64)

        # conv21
        self.conv21 = nn.Conv2d(64, 128, 3, padding= 1)
        self.bn21 = nn.BatchNorm2d(128)

        # conv22
        self.conv22 = nn.Conv2d(128, 128, 3, padding=1, stride=2)
        self.bn22 = nn.BatchNorm2d(128)

        # out
        self.fc = nn.Linear(128, 128)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = self.bn11(x)

        x = F.relu(self.conv12(x))
        x = self.bn12(x)

        x = F.relu(self.conv21(x))
        x = self.bn21(x)

        x = F.relu(self.conv22(x))
        x = self.bn22(x)

        x = F.avg_pool2d(x, kernel_size=[x.size(2), x.size(3)])
        x = self.fc(x.view(x.size()[:2]))#
        x = F.softmax(x)
        return x

class DeformConvNet(nn.Module):
    def __init__(self):
        super(DeformConvNet, self).__init__()
        
        # conv11
        self.conv11 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(32)

        # conv12
        self.offset12 = ConvOffset2D(32)
        self.conv12 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.bn12 = nn.BatchNorm2d(64)

        # conv21
        self.offset21 = ConvOffset2D(64)
        self.conv21 = nn.Conv2d(64, 128, 3, padding= 1)
        self.bn21 = nn.BatchNorm2d(128)

        # conv22
        self.offset22 = ConvOffset2D(128)
        self.conv22 = nn.Conv2d(128, 128, 3, padding=1, stride=2)
        self.bn22 = nn.BatchNorm2d(128)

        # out
        self.fc = nn.Linear(128, 128)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = self.bn11(x)
        
        x = self.offset12(x)
        x = F.relu(self.conv12(x))
        x = self.bn12(x)
        
        x = self.offset21(x)
        x = F.relu(self.conv21(x))
        x = self.bn21(x)
        
        x = self.offset22(x)
        x = F.relu(self.conv22(x))
        x = self.bn22(x)
        
        x = F.avg_pool2d(x, kernel_size=[x.size(2), x.size(3)])
        x = self.fc(x.view(x.size()[:2]))
        x = F.softmax(x)
        return x

    def freeze(self, module_classes):
        '''
        freeze modules for finetuning
        '''
        for k, m in self._modules.items():
            if any([type(m) == mc for mc in module_classes]):
                for param in m.parameters():
                    param.requires_grad = False

    def unfreeze(self, module_classes):
        '''
        unfreeze modules
        '''
        for k, m in self._modules.items():
            if any([isinstance(m, mc) for mc in module_classes]):
                for param in m.parameters():
                    param.requires_grad = True

    def parameters(self):
        return filter(lambda p: p.requires_grad, super(DeformConvNet, self).parameters())


from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNN(nn.Module):

    def __init__(self, vocab_size, embed_size, num_output, rnn_model='LSTM', use_last=True, embedding_tensor=None,
                 padding_index=0, hidden_size=64, num_layers=1, batch_first=True):
        """
        Args:
            vocab_size: vocab size
            embed_size: embedding size
            num_output: number of output (classes)
            rnn_model:  LSTM or GRU
            use_last:  bool
            embedding_tensor:
            padding_index:
            hidden_size: hidden size of rnn module
            num_layers:  number of layers in rnn module
            batch_first: batch first option
        """

        super(RNN, self).__init__()
        self.use_last = use_last
        # embedding
        self.encoder = None
        if torch.is_tensor(embedding_tensor):
            self.encoder = nn.Embedding(vocab_size, embed_size, padding_idx=padding_index, _weight=embedding_tensor)
            self.encoder.weight.requires_grad = False
        else:
            self.encoder = nn.Embedding(vocab_size, embed_size, padding_idx=padding_index)

        self.drop_en = nn.Dropout(p=0.6)

        # rnn module
        if rnn_model == 'LSTM':
            self.rnn = nn.LSTM( input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.5,
                                batch_first=True, bidirectional=True)
        elif rnn_model == 'GRU':
            self.rnn = nn.GRU( input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.5,
                                batch_first=True, bidirectional=True)
        else:
            raise LookupError('only support LSTM and GRU')


        self.bn2 = nn.BatchNorm1d(hidden_size*2)
        self.fc = nn.Linear(hidden_size*2, vocab_size)

    def forward(self, x, seq_lengths):
        '''
        Args:
            x: (batch, time_step, input_size)
        Returns:
            num_output size
        '''

        x_embed = self.encoder(x)
        x_embed = self.drop_en(x_embed)
        packed_input = pack_padded_sequence(x_embed, seq_lengths.cpu().numpy(),batch_first=True)

        # r_out shape (batch, time_step, output_size)
        # None is for initial hidden state
        packed_output, ht = self.rnn(packed_input, None)
        out_rnn, _ = pad_packed_sequence(packed_output, batch_first=True)

        row_indices = torch.arange(0, x.size(0)).long()
        col_indices = seq_lengths - 1
        if next(self.parameters()).is_cuda:
            row_indices = row_indices.cuda()
            col_indices = col_indices.cuda()

        if self.use_last:
            last_tensor=out_rnn[row_indices, col_indices, :]
        else:
            # use mean
            last_tensor = out_rnn[row_indices, :, :]
            last_tensor = torch.mean(last_tensor, dim=1)

        fc_input = self.bn2(last_tensor)
        out = self.fc(fc_input)
        return out


    
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
        # print(features.unsqueeze(0).unsqueeze(0))
        # print(features)
        # print(embeddings)
        embeddings = torch.cat((features.unsqueeze(0), embeddings), 1)
        # print(embeddings.transpose(0,1).size(1))
        # print()
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        hiddens, _ = self.lstm(packed)
        outputs = self.linear(hiddens[0])
        # print(lengths)
        # print(outputs)
        # exit(0)
        return outputs

    def sample(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(30):                                      # maximum sampling length
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
            outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            # print(np.asarray(outputs).shape)            
            # print(outputs)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
            inputs = inputs.unsqueeze(1)                         # (batch_size, 1, embed_size)
        # print(sampled_ids)
        sampled_ids = torch.cat(sampled_ids, 0)                  # (batch_size, 20)
        # print(sampled_ids)
        # exit(0)
        return sampled_ids.squeeze()

    def sample_beam_search(self, features, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(0)
        beam_size = 5
        candidates = []
        all_candidates = []
        for i in range(30):                                      # maximum sampling length
            if i==0:
                hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
                outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size) 
                predictions = torch.topk(outputs,beam_size)
                for k in range(beam_size):
                    candidates.append([predictions[1][0][k], predictions[0][0][k].cpu().data.numpy()[0] , hiddens , states])   
            else: 
                all_candidates = []
                for k in range(beam_size):
                    candidate = candidates[k]
                    inputs = self.embed(candidate[0][len(candidate[0])-1])
                    inputs = inputs.unsqueeze(0)
                    # print(inputs)
                    hiddens, states = self.lstm(inputs, candidate[3])          # (batch_size, 1, hidden_size), 
                    outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)        
                    predictions = torch.topk(outputs,beam_size)
                    for k in range(beam_size):
                        new_candidate = [torch.cat((candidate[0],predictions[1][0][k]),0),candidate[1] + predictions[0][0][k].cpu().data.numpy()[0], hiddens, states]
                        all_candidates.append(new_candidate)
                ordered = sorted(all_candidates, key=lambda tup:tup[1], reverse = True)
                candidates = ordered[:beam_size]
        sampled_ids = candidates[0][0]
        return sampled_ids.squeeze()

def get_cnn():
    return ConvNet()

def get_deform_cnn(trainable=True, freeze_filter=[nn.Conv2d, nn.Linear]):
    model = DeformConvNet()
    if not trainable:
        model.freeze(freeze_filter)
    return model