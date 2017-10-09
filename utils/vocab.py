class Vocab(object):

    """A vocabulary object."""

    def __init__(self, unk='<unk>', pad='<pad>', bos='<s>', eos='</s>'):
        self.unk = unk
        self.pad = pad
        self.bos = bos
        self.eos = eos
        self._itos = []
        if unk:
            self._itos.append(unk)
        if pad:
            self._itos.append(pad)
        if bos:
            self._itos.append(bos)
        if eos:
            self._itos.append(eos)
        self._stoi = {s: i for i, s in enumerate(self._itos)}

    def itos(self, i):
        return self._itos[i]

    def stoi(self, s):
        if self.unk:
            return self._stoi.get(s, self._stoi[self.unk])
        return self._stoi[s]
