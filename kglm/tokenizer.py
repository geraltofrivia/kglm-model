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


class SpacyTokenizer:
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
        self.nlp = spacy.load(spacy_lang, disable=['tagger', 'parser', 'ner'])
        if pretokenized:
            self.nlp.tokenizer = NullTokenizer(self.nlp.vocab)
        self.vocab = vocab

    def tokenize(self, text: str, keep_spacy_tokens: bool = False) -> List[Union[str, tokens.Token]]:
        """ Spacy Process the doc, and return spay tokens (or str) based on the flag"""
        doc = self.nlp(text)
        if keep_spacy_tokens:
            return [token for token in doc if not token.is_space]
        else:
            return [token.text for token in doc if not token.is_space]

    def batch_tokenize(self, texts: List[str], keep_spacy_tokens: bool = False) \
            -> List[List[Union[str, tokens.Token]]]:

        ops = []
        for text in texts:
            op = self.tokenize(text, keep_spacy_tokens=keep_spacy_tokens)
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
            texts: List[str],
            pad: bool = False,
            to: Optional[str] = None,
            max_len: int = -1
    ):
        if to not in ['torch', None, 'list']:
            raise BadParameters(f"Parameter to (: {to}) wants us to convert to some unknown data type.")

        # Convert to word tokens first
        tokenized = self.batch_tokenize(texts, keep_spacy_tokens=False)

        # Convert them to word IDs
        vocabularized = self.batch_vocabularize(tokenized)

        # Get pad lengths. Warning: may not be used
        biggest_seq = max(len(instance) for instance in vocabularized)

        if to == 'torch':
            # We WILL have to pad
            tensor = torch.zeros((len(vocabularized), biggest_seq), dtype=torch.int64)

            # Offset it with padding ID
            tensor += self.vocab[DEFAULT_PAD_TOKEN]

            # Put each instance on it
            for i, instance in enumerate(vocabularized):
                # TODO solve RunTimeError
                # RuntimeError: expand(torch.LongTensor{[10, 70]}, size=[70]): the number of sizes provided
                # (1) must be greater or equal to the number of dimensions in the tensor (2)
                tensor[i, :len(instance)] = tensor

            # If Max length is specified, cut things off there
            if max_len > 0 and max_len > biggest_seq:
                return tensor[:, :max_len]

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
