import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable


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
        embeddings = torch.cat((features.unsqueeze(0).unsqueeze(0), embeddings), 1)
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
        inputs = features.unsqueeze(0).unsqueeze(0)
        beam_size = 5
        candidates = []
        all_candidates = []
        for i in range(30):                                      # maximum sampling length
            if i==0:
                # print(inputs)
                hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
                outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size) 
                # softm = nn.Softmax(dim=len(outputs))
                # outputs = softm(outputs)
                predictions = torch.topk(outputs,beam_size)
                for k in range(beam_size):
                    candidates.append([predictions[1][0][k], predictions[0][0][k].cpu().data.numpy()[0] , hiddens , states])   
                    # print(predictions[1][0][k])
            else: 
                # print(i)
                all_candidates = []
                for k in range(beam_size):
                    # print(len(candidates))
                    candidate = candidates[k]
                    # print(candidate[0][0])
                    # print(len(candidate[0]))
                    # print(candidate[0])
                    inputs = self.embed(candidate[0][len(candidate[0])-1])
                    inputs = inputs.unsqueeze(1)
                    print(inputs)
                    hiddens, states = self.lstm(inputs, candidate[3])          # (batch_size, 1, hidden_size), 
                    outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)        
                    # softm = nn.Softmax()
                    # outputs = softm(outputs)
                    predictions = torch.topk(outputs,beam_size)
                    for k in range(beam_size):
                        # print(type(candidate[0]))
                        # print(torch.cat((candidate[0][0],predictions[1][0][k]),0))
                        # candidate[0].append(predictions[1][0][k])
                        # print(candidate[0])
                        # print(torch.cat((candidate[0],predictions[1][0][k]),0))
                        new_candidate = [torch.cat((candidate[0],predictions[1][0][k]),0),candidate[1] + predictions[0][0][k].cpu().data.numpy()[0], hiddens, states]
                        # print(len(candidate[0]))
                        # print(candidate[0])
                        # exit(0)
                        # print(torch.cat((candidate[0],predictions[1][0][k]),0))
                        all_candidates.append(new_candidate)
                # print(all_candidates[0][1])
                ordered = sorted(all_candidates, key=lambda tup:tup[1], reverse = True)
                candidates = ordered[:beam_size]
                ####SORT THE TOP 5 CANDIDATES OUT OF THE CANDIDATES IN ALL_CANDIDATES BASED ON THE LOG PROBABILITY
                ####STORE THESE TOP 5 INTO THE CANDIDATES VARIABLE AND START OVER
            # hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size), 
            # outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            # # print(np.asarray(outputs).shape)            
            # # print(outputs)
            # predicted = outputs.max(1)[1]
            # sampled_ids.append(predicted)
            # inputs = self.embed(predicted)
            # inputs = inputs.unsqueeze(1)                         # (batch_size, 1, embed_size)
        # print(sampled_ids)
        # print(candidates[0][0]) 
        sampled_ids = candidates[0][0]
        # sampled_ids = torch.cat(sampled_ids, 0)                  # (batch_size, 20)
        # print(sampled_ids)
        # exit(0)
        return sampled_ids.squeeze()
