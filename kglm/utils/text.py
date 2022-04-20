from typing import List
from allennlp.data.tokenizers import Tokenizer


def tokenize_to_string(text: str, tokenizer: Tokenizer) -> List[str]:
    """Sigh"""
    return [token.text for token in tokenizer.tokenize(text)]
