import torch

class Classification(torch.nn.Module):
    def __init__(self, idim=39, odim=1, hidden_dim=512, blstm_hidden_dim=128, num_layers=2):
        super(Classification, self).__init__()
        self.blstm = torch.nn.LSTM(input_size=512, hidden_size=blstm_hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.fc1 = torch.nn.Linear(idim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.relu = torch.nn.ReLU()
        self.fc4 = torch.nn.Linear(blstm_hidden_dim*2, odim)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        """
        Input:
            audio_feat: <tensor.FloatTensor> the audio features in a tensor
        Return:
            The predicted posterior probabilities
        """
        # print(x.shape)
        x = x.flatten(start_dim=2)
        # print(x.shape)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        # print(x.shape)

        x, _ = self.blstm(x)
        # print(x.shape)

        x = self.fc4(x)
        # batch_size, sequence_length, num_features, context_window = x.size()
        # x = x.view(batch_size, sequence_length, -1)
        x = x.transpose(1, 2)
        return x