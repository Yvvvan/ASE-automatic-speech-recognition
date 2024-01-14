import torch


class Classification(torch.nn.Module):
    def __init__(self, idim, odim, hidden_dim=512):
        torch.nn.Module.__init__(self)
        # Define three fully connected layers followed by ReLU activation funtion
        self.fc1 = torch.nn.Linear(idim, hidden_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = torch.nn.ReLU()
        self.classification_layer = torch.nn.Linear(hidden_dim, odim)

    def forward(self, audio_feat):
        """
        Input: 
            audio_feat: <tensor.FloatTensor> the audio features in a tensor
        Return: 
            The predicted posterior probabilities
        """
        # flatten  the input tensor [BS, f len, f dim, c dim] -> [BS, f len, f dim * c dim]
        x = audio_feat.flatten(start_dim=2)
        # Features mapped by 3 fully connected layers
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.classification_layer(x)
        # swap the dimension of the output tensor [BS, f len, odim] -> [BS, odim, f len]
        x = x.transpose(1, 2)
        return x

