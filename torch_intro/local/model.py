import torch


class Classification(torch.nn.Module):
    def __init__(self, idim=39, odim=1, hidden_dim=512):
        torch.nn.Module.__init__(self)
        # Define three fully connected layers followed by ReLU activation funtion
        self.fc1 = torch.nn.Linear(idim, hidden_dim)
        self.relu1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu3 = torch.nn.ReLU()
        # Define a fully connected output (classification) layer
        self.fc4 = torch.nn.Linear(hidden_dim, odim)
        # Sigmoid function computes probability
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, audio_feat):
        """
        Input: 
            audio_feat: <tensor.FloatTensor> the audio features in a tensor
        Return: 
            The predicted posterior probabilities
        """
        # Features mapped by 3 fully connected layers
        x = self.fc1(audio_feat)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        # Average the mapped representation over the sequence dimension
        x = torch.mean(x, dim=1)
        # Return the predicted probabilities by classification layer and Sigmoid function.
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x

