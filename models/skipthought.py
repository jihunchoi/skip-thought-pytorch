from torch import nn

from models import encoders, decoders


class SkipThought(nn.Module):

    def __init__(self, rnn_type, num_words, word_dim, hidden_dim,
                 bidirectional, predict_prev, predict_cur, predict_next):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_words = num_words
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        self.predict_prev = predict_prev
        self.predict_cur = predict_cur
        self.predict_next = predict_next

        encoder_hidden_dim = hidden_dim
        if bidirectional:
            encoder_hidden_dim = hidden_dim // 2
        self.encoder = encoders.RecurrentEncoder(
            rnn_type=rnn_type, num_words=num_words, word_dim=word_dim,
            hidden_dim=encoder_hidden_dim, dropout_prob=0,
            bidirectional=bidirectional, num_layers=1)

        def make_decoder():
            return decoders.RecurrentDecoder(
                rnn_type=rnn_type, num_words=num_words, word_dim=word_dim,
                hidden_dim=hidden_dim, num_layers=1, dropout_prob=0)

        if predict_prev:
            setattr(self, 'decoder_prev', make_decoder())
        if predict_cur:
            setattr(self, 'decoder_cur', make_decoder())
        if predict_next:
            setattr(self, 'decoder_next', make_decoder())

        self.reset_parameters()

    def get_decoder(self, name):
        return getattr(self, f'decoder_{name}')

    def reset_parameters(self):
        self.encoder.reset_parameters()
        if self.predict_prev:
            self.get_decoder('prev').reset_parameters()
        if self.predict_cur:
            self.get_decoder('cur').reset_parameters()
        if self.predict_next:
            self.get_decoder('next').reset_parameters()

    def forward(self, src, tgt):
        _, encoder_state = self.encoder(words=src[0], length=src[1])
        if isinstance(encoder_state, tuple):  # LSTM
            encoder_state = encoder_state[0]
        if self.bidirectional:
            context = (encoder_state.transpose(0, 1).contiguous()
                       .view(-1, self.hidden_dim))
        else:
            context = encoder_state.squeeze(0)
        logits = dict()
        for k in tgt:
            logits_k, _ = self.get_decoder(k)(
                words=tgt[k][0], prev_state=context.unsqueeze(0))
            logits[k] = logits_k
        return logits
