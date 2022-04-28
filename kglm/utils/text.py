from typing import List, Iterable
from allennlp.data.tokenizers import Tokenizer, Token



def tokenize(iterable: Iterable[str]):
    """
    Used inside ~/datareaders
    TODO: figure out what it does precisely and write docstrings
    """
    return [Token(x) for x in iterable]
