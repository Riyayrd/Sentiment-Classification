# src/models.py
import torch
import torch.nn as nn

class SentimentRNN(nn.Module):
    """
    model_type: 'RNN', 'LSTM', 'BiLSTM'
    activation: 'relu','tanh','sigmoid' (applied to classifier dense layer)
    num_layers: number of recurrent layers (assignment expects 2)
    hidden_size: hidden dimension per direction
    """
    def __init__(self,
                 model_type='LSTM',
                 vocab_size=10000,
                 embed_dim=100,
                 hidden_size=64,
                 num_layers=2,
                 dropout=0.4,
                 activation='tanh'):
        super().__init__()
        self.model_type = model_type
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        bidirectional = (model_type == 'BiLSTM')
        if model_type == 'RNN':
            # nn.RNN supports nonlinearity param 'tanh' or 'relu'
            nonlin = 'tanh' if activation == 'tanh' else 'relu'
            self.rnn = nn.RNN(embed_dim,
                              hidden_size,
                              num_layers=num_layers,
                              nonlinearity=nonlin,
                              batch_first=True,
                              dropout=dropout,
                              bidirectional=bidirectional)
        elif model_type == 'LSTM' or model_type == 'BiLSTM':
            self.rnn = nn.LSTM(embed_dim,
                               hidden_size,
                               num_layers=num_layers,
                               batch_first=True,
                               dropout=dropout,
                               bidirectional=bidirectional)
        else:
            raise ValueError("model_type must be 'RNN','LSTM' or 'BiLSTM'")

        self.rnn_output_size = hidden_size * (2 if bidirectional else 1)

        # classifier (two dense layers as required: fc1 -> fc2)
        self.dropout = nn.Dropout(dropout)
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError("activation must be 'relu','tanh','sigmoid'")

        self.fc1 = nn.Linear(self.rnn_output_size, 64)  # hidden dense
        self.fc2 = nn.Linear(64, 1)  # output dense
        self.out_act = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)  # (b, seq, embed)
        rnn_out, _ = self.rnn(emb)  # rnn_out: (b, seq, hidden*dirs)
        # Mean pooling over time dimension
        pooled = rnn_out.mean(dim=1)
        h = self.dropout(pooled)
        h = self.activation(self.fc1(h))
        h = self.dropout(h)
        logits = self.fc2(h)
        out = self.out_act(logits).squeeze(-1)
        return out
