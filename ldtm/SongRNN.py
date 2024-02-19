import math

import torch
import torch.nn as nn


class SongRNN(nn.Module):
    def __init__(self, input_size, output_size, config):
        """
        Initialize the SongRNN model.
        """
        super(SongRNN, self).__init__()

        embedding_dim = config["hidden_size"]
        NUM_LAYERS = config["no_layers"]
        MODEL_TYPE = config["model_type"].upper()
        DROPOUT_P = config["dropout"]

        self.model_type = MODEL_TYPE
        self.input_size = input_size
        self.embedding_dim = embedding_dim
        self.output_size = output_size
        self.num_layers = NUM_LAYERS
        self.dropout_p = DROPOUT_P

        # Initialize embedding layer
        self.embedding = nn.Embedding(input_size, embedding_dim)

        # Initialize recurrent layer (LSTM or RNN)
        self.lstm: nn.LSTM | None
        self.rnn: nn.RNN | None
        if MODEL_TYPE == "LSTM":
            self.lstm = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=embedding_dim,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT_P,
                batch_first=True,
            )
        elif MODEL_TYPE == "RNN":
            self.rnn = nn.RNN(
                input_size=embedding_dim,
                hidden_size=embedding_dim,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT_P,
                batch_first=True,
            )
        else:
            raise ValueError(
                "Invalid model type. Supported types are 'LSTM' and 'RNN'."
            )

        # Initialize linear output layer
        self.fc = nn.Linear(embedding_dim, output_size)

        # Initialize dropout layer
        self.dropout = nn.Dropout(DROPOUT_P)
        self.init_weights()

    def init_weights(self):
        init_range_emb = 0.1
        init_range_other = 1 / math.sqrt(self.embedding_dim)
        self.embedding.weight.data.uniform_(-init_range_emb, init_range_emb)
        self.fc.weight.data.uniform_(-init_range_other, init_range_other)
        self.fc.bias.data.zero_()
        if self.model_type == "LSTM":
            assert self.lstm is not None
            for i in range(self.num_layers):
                self.lstm.all_weights[i][0] = torch.FloatTensor(  # type: ignore
                    self.embedding_dim, self.embedding_dim
                ).uniform_(-init_range_other, init_range_other)
                self.lstm.all_weights[i][1] = torch.FloatTensor(  # type: ignore
                    self.embedding_dim, self.embedding_dim
                ).uniform_(-init_range_other, init_range_other)
        else:
            raise NotImplementedError

    def init_hidden(
        self, batch_size, device
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Initializes the hidden state for the recurrent neural network.

        TODO:
        ----
        Check the model type to determine the initialization method:
        (i) If model_type is LSTM, initialize both cell state and hidden state.
        (ii) If model_type is RNN, initialize the hidden state only.

        Initialise with zeros.
        """
        if self.model_type == "LSTM":
            return (
                torch.zeros(self.num_layers, batch_size, self.embedding_dim).to(device),
                torch.zeros(self.num_layers, batch_size, self.embedding_dim).to(device),
            )
        elif self.model_type == "RNN":
            return torch.zeros(self.num_layers, batch_size, self.embedding_dim).to(
                device
            ), None
        else:
            raise ValueError(
                "Invalid model type. Supported types are 'LSTM' and 'RNN'."
            )

    def detach_hidden(self, hidden):
        hidden, cell = hidden
        hidden = hidden.detach()
        cell = cell.detach()
        return hidden, cell

    def forward(self, seq, hidden):
        """
        Forward pass of the SongRNN model.
        (Hint: In teacher forcing, for each run of the model, input will be a single character
        and output will be pred-probability vector for the next character.)

        Returns:
        -------
        - output (Tensor): Output tensor of shape (output_size)
        - activations (Tensor): Hidden layer activations to plot heatmap values

        TODOs:
        (i) Embed the input sequence
        (ii) Forward pass through the recurrent layer
        (iii) Apply dropout (if needed)
        (iv) Pass through the linear output layer
        """

        embedded = self.embedding(seq)
        if self.model_type == "LSTM":
            # hidden = (hidden[0].squeeze(), hidden[1].squeeze())
            output, hidden = self.lstm(embedded, hidden)
        elif self.model_type == "RNN":
            output, _ = self.rnn(embedded)
            output = output[:, -1, :]  # Take the last time step's output
            raise NotImplementedError
        else:
            raise Exception(f"Invalid model type {self.model_type}")

        if self.training:
            output = self.dropout(output)
        output = self.fc(output)

        return output, hidden
