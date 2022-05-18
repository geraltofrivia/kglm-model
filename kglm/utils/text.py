from spacy.tokens import Doc
try:
    from spacy.util import DummyTokenizer
except ImportError:
    # noinspection PyUnusedLocal,PyMethodMayBeStatic
    class DummyTokenizer:
        def __call__(self, text):
            raise NotImplementedError

        def pipe(self, texts, **kwargs):
            for text in texts:
                yield self(text)

        # add dummy methods for to_bytes, from_bytes, to_disk and from_disk to
        # allow serialization (see #1557)
        def to_bytes(self, **kwargs):
            return b""

        def from_bytes(self, _bytes_data, **kwargs):
            return self

        def to_disk(self, _path, **kwargs):
            return None

        def from_disk(self, _path, **kwargs):
            return self


class NullTokenizer(DummyTokenizer):
    """
    Use it when the text is already tokenized but the doc's gotta go through spacy.
    Usage: `nlp.tokenizer = CustomTokenizer(nlp.vocab)`
    """

    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, words):
        return Doc(self.vocab, words=words)

