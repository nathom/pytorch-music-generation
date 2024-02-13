import torch
import torch.nn as nn


class SongRNN(nn.Module):
    def __init__(self, input_size, output_size, config):
        """
        Initialize the SongRNN model.
        """
        super(SongRNN, self).__init__()

        HIDDEN_SIZE = config["hidden_size"]
        NUM_LAYERS = config["no_layers"]
        MODEL_TYPE = config["model_type"]
        DROPOUT_P = config["dropout"]

        self.model_type = MODEL_TYPE
        self.input_size = input_size
        self.hidden_size = HIDDEN_SIZE
        self.output_size = output_size
        self.num_layers = NUM_LAYERS
        self.dropout_p = DROPOUT_P

        # Initialize embedding layer
        self.embedding = nn.Embedding(input_size, HIDDEN_SIZE)

        # Initialize recurrent layer (LSTM or RNN)
        if MODEL_TYPE == "LSTM":
            self.rnn = nn.LSTM(
                input_size=HIDDEN_SIZE,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT_P,
                batch_first=True,
            )
        elif MODEL_TYPE == "RNN":
            self.rnn = nn.RNN(
                input_size=HIDDEN_SIZE,
                hidden_size=HIDDEN_SIZE,
                num_layers=NUM_LAYERS,
                dropout=DROPOUT_P,
                batch_first=True,
            )
        else:
            raise ValueError(
                "Invalid model type. Supported types are 'LSTM' and 'RNN'."
            )

        # Initialize linear output layer
        self.linear = nn.Linear(HIDDEN_SIZE, output_size)

        # Initialize dropout layer
        self.dropout = nn.Dropout(DROPOUT_P)

    def init_hidden(self, batch_size):
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
                torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size),
            )
        elif self.model_type == "RNN":
            return torch.zeros(self.num_layers, batch_size, self.hidden_size)
        else:
            raise ValueError(
                "Invalid model type. Supported types are 'LSTM' and 'RNN'."
            )

    def forward(self, seq):
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
        rnn_out, _ = self.rnn(embedded)
        rnn_out = rnn_out[:, -1, :]  # Take the last time step's output
        dropped_out = self.dropout(rnn_out)
        output = self.linear(dropped_out)

        return output
