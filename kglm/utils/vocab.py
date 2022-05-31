"""
    File containing classes which act as glorified dictionaries.
    Lightweight, though.
"""
from pathlib import Path
from typing import Union, Dict, Optional, List

import torch

from config import DEFAULT_UNK_TOKEN, DEFAULT_PAD_TOKEN, DEFAULT_BOS_TOKEN, DEFAULT_EOS_TOKEN, LOCATIONS as LOC


class Vocab:

    def __init__(
            self,
            vocab: List[str],
            filename: Optional[Union[str, Path]],
            skip_eos: bool = False,
            skip_pad: bool = False,
            skip_bos: bool = False,
            skip_unk: bool = False,
            name: str = None):

        # Store some metadata
        self._name: str = name if name is not None else ''
        self._filename: Optional[Path] = Path(filename) if filename is not None else None

        self._skip_pad = skip_pad
        self._skip_unk = skip_unk
        self._skip_bos = skip_bos
        self._skip_eos = skip_eos

        # Check if padding token appears normally in the vocab. If not, prepend it to the vocab
        # TODO: may be the vocab should not be forcefully prepended. Check down the line if changing this helps

        specials = []

        if DEFAULT_PAD_TOKEN not in vocab and not skip_pad:
            specials += [DEFAULT_PAD_TOKEN]

        if DEFAULT_UNK_TOKEN not in vocab and not skip_unk:
            specials += [DEFAULT_UNK_TOKEN]

        if DEFAULT_BOS_TOKEN not in vocab and not skip_bos:
            specials += [DEFAULT_BOS_TOKEN]

        if DEFAULT_EOS_TOKEN not in vocab and not skip_eos:
            specials += [DEFAULT_EOS_TOKEN]

        vocab = specials + vocab

        self.tok_to_id: Dict[str, int] = {tok: i for i, tok in enumerate(vocab)}
        self.id_to_tok: List[str] = vocab

        print(f"A vocab: {name} with {len(self.tok_to_id)} elements ready to go!")

    @property
    def pad(self):
        try:
            return self.tok_to_id[DEFAULT_PAD_TOKEN]
        except KeyError as e:
            if self._skip_pad:
                raise ValueError("We were specified to not make way for the pad token")
            else:
                raise e

    @property
    def unk(self):
        try:
            return self.tok_to_id[DEFAULT_UNK_TOKEN]
        except KeyError as e:
            if self._skip_pad:
                raise ValueError("We were specified to not make way for the unk token")
            else:
                raise e

    @classmethod
    def load(cls, file: Path, name: Optional[str] = None, *args, **kwargs):
        """ A generator which returns a Vocab object. If name is not provided, it is inferred."""

        name = file.name if name is None else name

        with file.open('r') as f:
            vocab = [x for x in f.read().splitlines() if x.strip() != '']

        return cls(*args, name=name, filename=file, vocab=vocab, **kwargs)

    def __getitem__(self, item):
        return self.tok_to_id[item]

    def get(self, item, alternate):
        return self.tok_to_id.get(item, alternate)

    def __len__(self):
        return len(self.tok_to_id)

    def _encode_token_(self, token: str, allow_unknowns: bool) -> int:
        """ just a 'get' with a unknown fallback if needed """
        if not allow_unknowns:
            return self[token]

        return self.get(token, self.unk)

    def encode(self, seq: Union[str, List[str]], allow_unknowns: bool = True, return_type: Optional[str] = None):
        if type(seq) is str:
            ids = self._encode_token_(seq, allow_unknowns=allow_unknowns)
        else:
            # Its an actual sequence
            ids = [self._encode_token_(token, allow_unknowns=allow_unknowns) for token in seq]

        if return_type == 'torch':
            return torch.tensor(ids, dtype=torch.int64)
        elif return_type in [None, 'list']:
            return ids

    def _decode_token_(self, tok_id: int):
        """ hard fails. not graceful """
        return self.id_to_tok[tok_id]

    def decode(self, tok_ids: List[int]):
        # TODO allow it to handle tensors also
        return [self.id_to_tok[tok_id] for tok_id in tok_ids]


if __name__ == '__main__':

    vocab_fdir = LOC.vocab / 'tokens.txt'
    vocab = Vocab.load(vocab_fdir)
    print(vocab.encode("red"))
    sent = ["red", "tomatoes", "grow", "on", "a", "supercalifragiliciousexpialidocius", "wine", "."]

    print(vocab.encode(sent))
    encoded = vocab.encode(sent)
    print(vocab.decode(encoded))
    print(len(vocab))

    all_ids = list(vocab.tok_to_id.values())
    tensor = torch.LongTensor(all_ids)
