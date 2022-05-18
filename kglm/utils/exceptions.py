class ConfigurationError(Exception):
    """ To be raised when we get weird config combos"""
    ...


class BadParameters(ValueError):
    """ Raise it when you don't like a parameter being invoked
        but are too lazy to create custom data type to restrict its values. """
    ...


class NoVocabInTokenizer(AttributeError):
    """ Raise when someone uses tokenizer to vocabularise things but didnt' provide a vocab to it. """
    ...