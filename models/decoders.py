from torch import nn
from torch.autograd import Variable
from torch.nn import init


class RecurrentDecoder(nn.Module):

    def __init__(self, rnn_type, num_words, word_dim, hidden_dim,
                 num_layers, dropout_prob):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_words = num_words
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout_prob = dropout_prob

        self.dropout = nn.Dropout(dropout_prob)
        self.word_embedding = nn.Embedding(num_embeddings=num_words,
                                           embedding_dim=word_dim)
        if rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=word_dim, hidden_size=hidden_dim,
                num_layers=num_layers, dropout=dropout_prob)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=word_dim, hidden_size=hidden_dim,
                num_layers=num_layers, dropout=dropout_prob)
        else:
            raise ValueError('Unknown RNN type!')
        self.output_linear = nn.Linear(in_features=hidden_dim,
                                       out_features=num_words)
        self.reset_parameters()

    def reset_parameters(self):
        init.normal(self.word_embedding.weight.data, mean=0, std=0.01)
        for i in range(self.num_layers):
            weight_ih = getattr(self.rnn, f'weight_ih_l{i}')
            weight_hh = getattr(self.rnn, f'weight_hh_l{i}')
            bias_ih = getattr(self.rnn, f'bias_ih_l{i}')
            bias_hh = getattr(self.rnn, f'bias_hh_l{i}')
            init.orthogonal(weight_hh.data)
            init.kaiming_normal(weight_ih.data)
            init.constant(bias_ih.data, val=0)
            init.constant(bias_hh.data, val=0)
            if self.rnn_type == 'lstm':  # Set initial forget bias to 1
                bias_ih.data.chunk(4)[1].fill_(1)
        init.kaiming_normal(self.output_linear.weight.data)
        init.constant(self.output_linear.bias.data, val=0)

    def forward(self, words, prev_state):
        """
        Args:
            words (Variable): A long variable of size
                (length, batch_size) that contains indices of words.
            prev_state (Variable): The previous state of the decoder.

        Returns:
            logits (Variable): A float variable containing unnormalized
                log probabilities. It has the same size as words.
            state (Variable): The current state of the decoder.
        """

        words_emb = self.word_embedding(words)
        words_emb = self.dropout(words_emb)
        rnn_outputs, rnn_state = self.rnn(input=words_emb, hx=prev_state)
        logits = self.output_linear(rnn_outputs)
        return logits, rnn_state
