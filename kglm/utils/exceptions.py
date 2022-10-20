class ConfigurationError(Exception):
    """
    The exception raised by any AllenNLP object when it's misconfigured
    (e.g. missing properties, invalid properties, unknown properties).
    """

    def __init__(self, message):
        super(ConfigurationError, self).__init__()
        self.message = message

    def __str__(self):
        return repr(self.message)


class BadParameters(ValueError):
    """ Raise it when you don't like a parameter being invoked
        but are too lazy to create custom data type to restrict its values. """
    ...


class NoVocabInTokenizer(AttributeError):
    """ Raise when someone uses tokenizer to vocabularise things but didnt' provide a vocab to it. """
    ...


class FoundNaNs(ValueError):
    """ If you ever encounter nans in the model, raise this, why don't you.  """
    ...

