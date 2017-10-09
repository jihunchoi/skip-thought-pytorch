import os
import random

import torch


class DataReader(object):

    """A data reader that iterates over multiple shards."""

    def __init__(self, data_dir, shuffle=True):
        self.file_paths = [os.path.join(data_dir, filename)
                           for filename in sorted(os.listdir(data_dir))]
        self.files = [open(file_path, 'r', encoding='utf-8')
                      for file_path in self.file_paths]
        self.shuffle = shuffle
        self._buffer = []

    def start_epoch(self):
        self.files = [open(file_path, 'r', encoding='utf-8')
                      for file_path in self.file_paths]
        del self._buffer[:]

    def _read_next_triple(self):
        if not self.files:
            raise StopIteration
        if self.shuffle:
            file_idx = random.randrange(len(self.files))
        else:
            file_idx = 0
        # Read three consecutive lines, but advance for only one line.
        file = self.files[file_idx]
        prev = file.readline().strip()
        pos = file.tell()
        cur = file.readline().strip()
        next_ = file.readline().strip()
        file.seek(pos)
        if prev and cur and next_:
            return prev, cur, next_
        else:
            # A blank line means that the file reached EOF
            del self.files[file_idx]
            return self._read_next_triple()

    def _fill_buffer(self, buffer_size):
        fill_size = buffer_size - len(self._buffer)
        for _ in range(fill_size):
            try:
                self._buffer.append(self._read_next_triple())
            except StopIteration:
                break

    def _ensure_buffer_filled(self, min_size, buffer_size):
        if len(self._buffer) > min_size:
            return
        self._fill_buffer(buffer_size)

    def _read_from_buffer(self, size, peek=False):
        self._ensure_buffer_filled(min_size=size, buffer_size=10000)
        data = self._buffer[:size]
        if not peek:
            del self._buffer[:size]
        return data

    def next_batch(self, size, peek=False):
        data = self._read_from_buffer(size=size, peek=peek)
        if not data:
            raise StopIteration
        return data


class Preprocessor(object):

    """Preprocess data triples in a pretty form."""

    def __init__(self, predict_prev, predict_cur, predict_next, vocab,
                 split_fn=str.split):
        self.predict_prev = predict_prev
        self.predict_cur = predict_cur
        self.predict_next = predict_next
        self.vocab = vocab
        self.split_fn = split_fn

    def _split_into_words(self, batch):
        return [self.split_fn(d) for d in batch]

    def _wrap_with_bos_eos(self, batch):
        return [[self.vocab.bos] + d + [self.vocab.eos]
                for d in batch]

    def _pad_batch(self, batch):
        lengths = [len(d) for d in batch]
        max_length = max(lengths)
        padded_batch = [d + [self.vocab.pad]*(max_length - len(d))
                        for d in batch]
        return padded_batch, lengths

    def _numericalize(self, batch):
        return [[self.vocab.stoi(s) for s in d]
                for d in batch]

    def _process_src(self, batch):
        padded_batch, lengths = self._pad_batch(batch)
        return self._numericalize(padded_batch), lengths

    def _process_tgt(self, batch):
        padded_batch, lengths = self._pad_batch(self._wrap_with_bos_eos(batch))
        return self._numericalize(padded_batch), lengths

    def __call__(self, batch):
        prev, cur, next_ = list(zip(*batch))
        prev = self._split_into_words(prev)
        cur = self._split_into_words(cur)
        next_ = self._split_into_words(next_)

        src = self._process_src(cur)
        tgt = dict()
        if self.predict_prev:
            tgt['prev'] = self._process_tgt(prev)
        if self.predict_cur:
            tgt['cur'] = self._process_tgt(cur)
        if self.predict_next:
            tgt['next'] = self._process_tgt(next_)
        src = (torch.LongTensor(src[0]), torch.LongTensor(src[1]))
        tgt = {k: (torch.LongTensor(v[0]), torch.LongTensor(v[1]))
               for k, v in tgt.items()}
        return {'src': src, 'tgt': tgt}
