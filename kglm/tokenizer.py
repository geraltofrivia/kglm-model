"""
    We're making stand alone tokenizers right now, in our crusade for de-allennlp-ifying the code base but we should
        aim to let go of this class down the line.

    This one just simply uses spaCy to tokenize text.
"""
import spacy
import torch
from spacy import tokens
from typing import List, Union, Optional

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from utils.vocab import Vocab
from utils.exceptions import BadParameters, NoVocabInTokenizer
from config import DEFAULT_UNK_TOKEN, DEFAULT_PAD_TOKEN
from utils.text import NullTokenizer
from abc import ABC, abstractmethod


class Tokenizer(ABC):

    def __init__(self, vocab: Optional[Vocab], pretokenized: bool):
        self.vocab: Vocab = vocab
        self._pretokenized: bool = pretokenized

    @abstractmethod
    def tokenize(self, text: Union[str, List[str]]) -> List[str]:
        ...

    def batch_tokenize(self, texts: List[Union[str, List[str]]]) -> List[List[str]]:

        ops = []
        for text in texts:
            op = self.tokenize(text)
            ops.append(op)

        return ops

    def vocabularize(self, tokens: List[str]) -> List[int]:

        if self.vocab is None:
            raise NoVocabInTokenizer

        return [self.vocab.get(token, self.vocab[DEFAULT_UNK_TOKEN]) for token in tokens]

    def batch_vocabularize(self, batch: List[List[str]]) -> List[List[int]]:
        return [self.vocabularize(text) for text in batch]

    def batch_convert(
            self,
            texts: List[Union[str, List[str]]],
            pad: bool = False,
            to: Optional[str] = None,
            max_len: int = -1
    ):
        """
            Expects 2D inputs (as can be guessed from the type hints
        Parameters
        ----------
        texts
        pad
        to
        max_len

        Returns
        -------

        """
        if to not in ['torch', None, 'list']:
            raise BadParameters(f"Parameter to (: {to}) wants us to convert to some unknown data type.")

        # Convert to word tokens first
        tokenized = self.batch_tokenize(texts)

        # Convert them to word IDs
        vocabularized = self.batch_vocabularize(tokenized)

        # Get pad lengths. Warning: may not be used
        biggest_seq = max(len(instance) for instance in vocabularized)

        if to == 'torch':
            # We WILL have to pad
            tensor = torch.zeros((len(vocabularized), biggest_seq), dtype=torch.int64)

            pad_token = self.vocab[DEFAULT_PAD_TOKEN]

            # Offset it with padding ID
            tensor += pad_token

            # Put each instance on it
            for i, instance in enumerate(vocabularized):
                # TODO solve RunTimeError
                # RuntimeError: expand(torch.LongTensor{[10, 70]}, size=[70]): the number of sizes provided
                # (1) must be greater or equal to the number of dimensions in the tensor (2)
                tensor[i, :len(instance)] = torch.tensor(instance)

            if 0 < max_len < biggest_seq:
                # If Max length is specified, and its smaller than current tensor, cut things off there
                return tensor[:, :max_len]
            elif 0 < biggest_seq < max_len:
                # If max len is specified and its bigger than current tensor, pad the tensor
                return torch.nn.functional.pad(tensor, (0, max_len-biggest_seq), "constant", pad_token)
            else:
                return tensor

        elif to in [None, 'list']:

            padding_id = self.vocab[DEFAULT_PAD_TOKEN]

            # Trim things to max len
            if max_len > 0:
                biggest_seq = min(biggest_seq, max_len)

            if pad:
                vocabularized = [instance[: biggest_seq] for instance in vocabularized]

                # At this point we have trimmed excesses wherever needed. Now we just need to pad wherever needed
                for i, instance in enumerate(vocabularized):
                    if len(instance) < biggest_seq:
                        vocabularized[i] = instance + [DEFAULT_PAD_TOKEN] * (biggest_seq - len(instance))

            return vocabularized

        else:

            raise Exception("Code should never have come here.")

    def batch_convert_3d(
            self,
            texts,
            pad: bool = True,
            to: Optional[str] = None
    ) -> torch.Tensor:
        """
            Similar to batch convert but for 3D tensors
        Parameters
        ----------
        texts
        pad
        to

        Returns
        -------

        """

        if to != 'torch' or not pad:
            raise NotImplementedError(f"Can not run this function for pad={pad}, to={to}")

        max_depth = max(max(len(x) for x in instance) for instance in texts)
        outputs = []
        for instance in texts:
            outputs.append(self.batch_convert(texts, pad=True, to='torch', max_len=max_depth))

        return torch.stack(outputs, dim=0)


class SimpleTokenizer(Tokenizer):
    """ A simpler instance of above where the text is already tokenized."""

    def __init__(self, vocab: Vocab):
        # We hardcode pretokenized is true
        super().__init__(vocab=vocab, pretokenized=True)

    def tokenize(self, text: List[str], *args, **kwargs) -> List[str]:
        return text


class SpacyTokenizer(Tokenizer):
    """ A simple obj wrapping a spacy NLP object, yields list of str or tokens as instructed by flag."""

    def __init__(self, vocab: Optional[Vocab] = None, spacy_lang: str = 'en_core_web_sm', pretokenized: bool = False):
        """
            A tokenizer which either gets text sequences (tokenized already or not) and it does a whole bunch of things
                over them including tokenization, converting str to IDs, padding them, converting them to tensor

            Alternatively, it can do the opposite ie give it a int or a list of ints and it tells us what is their text

        Parameters
        ----------
        vocab
        spacy_lang
        pretokenized
        """

        super().__init__(vocab=vocab, pretokenized=pretokenized)
        self.nlp = spacy.load(spacy_lang, disable=['tagger', 'parser', 'ner'])
        if pretokenized:
            self.nlp.tokenizer = NullTokenizer(self.nlp.vocab)

    def tokenize(self, text: Union[str, List[str]]) -> List[Union[str, tokens.Token]]:
        """ Spacy Process the doc, and return spay tokens (or str) based on the flag"""
        doc = self.nlp(text)
        return [token.text for token in doc if not token.is_space]

