# AI Rationalization Symbolic Representation
Code for AI rationalization using a symbolic game representation passed through a seq2seq instead of a CNN-RNN autoencoder. 

#### Training phase
The code has the following requirements, 
1. Python 3.6
2. Pytorch 0.4.0
3. Torchvision 0.2.1
4. nltk 3.3.0

To train the code run the following command from the root directory. 

python train_frogger_seq2seq.py

#### Test phase
To see the results of all the testing samples when passed through the network run, 
python sample_v6.py
