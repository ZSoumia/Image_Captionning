import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=3):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embed_size
        self.vocab_size = vocab_size
        
        self.embed_vect = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, vocab_size)
       
     
       
    
    def forward(self, features, captions):
        batch_size = features.size(0)
        
        caption = captions[:, :-1] # remove the end <end>
        captions = self.embed_vect(caption)
        
        # Concatenate the features and caption inputs and feed to LSTM cell(s).
        features = features.unsqueeze(1)
        inputs = torch.cat((features, captions), 1)
        lstm_output, _ = self.lstm(inputs, None)
       
        return self.fc(lstm_output)

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        i = 0
        caption = []
        word_code = None
        # We loop to get the caption while we haven't reached the max caption length or we reached <end> which has the code of 1
        while( (i < max_len) and(word_code != 1)) :
            # get the predicted output 
            output_lstm, states = self.lstm(inputs, states)
            output = self.fc(output_lstm)
            proba, word = output.max(2)
            #add the word to the caption
            caption.append(word.item())
            
            # update the next pass inputs
            inputs =  self.embed_vect(word)
              
            i = i +1
            
            
        return caption