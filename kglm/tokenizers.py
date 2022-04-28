"""
    We're making stand alone tokenizers right now, in our crusade for de-allennlp-ifying the code base but we should
        aim to let go of this class down the line.

    This one just simply uses spaCy to tokenize text.
"""
import spacy
from spacy import tokens
from typing import List, Union


class SpacyTokenizer:
    """ A simple obj wrapping a spacy NLP object, yields list of str or tokens as instructed by flag."""

    def __init__(self, spacy_model_nm: str = 'en_core_web_sm'):
        self.nlp = spacy.load(spacy_model_nm)

    def tokenize(self, text: str, keep_spacy_tokens: bool = False) -> List[Union[str, tokens.Token]]:
        """ Spacy Process the doc, and return spay tokens (or str) based on the flag"""
        doc = self.nlp(text)
        if keep_spacy_tokens:
            return [token for token in doc if not token.is_space]
        else:
            return [token.text for token in doc if not token.is_space]
