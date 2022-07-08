import torch
import pickle
import logging
import numpy as np
from tqdm.auto import tqdm
from collections import defaultdict
from typing import Dict, Set, Any, List, Tuple

# Allennlp imports
# TODO: get rid of it later.
# from allennlp.data import Vocabulary
from utils.vocab import Vocab
from tokenizer import SpacyTokenizer


# Local Imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from config import DEFAULTS


logger = logging.getLogger(__name__)
AliasList = List[List[str]]

"""
    _token_lookup is a dict like 
    {'Q31': [['Belgium'], ['Kingdom', 'of', 'Belgium'], ['be'], ['🇧', '🇪']],
     'Q23': [['George', 'Washington'],
      ['Washington'],
      ['President', 'Washington'],
      ['G.', 'Washington']], ... }

      To access it do:
      ad = ds._alias_database.load("data/linked-wikitext-2/alias.pkl")
      ad._token_lookup
"""


class AliasDatabase:
    """
        A Database of Aliases for entities, relations.
        Used extensively during data reading, and even in model forward.

        There's some preproc involved.

    """

    def __init__(self,
                 token_lookup: Dict[str, AliasList],
                 id_map_lookup: Dict[str, Dict[str, int]],
                 id_array_lookup: Dict[str, np.ndarray],
                 token_to_entity_lookup: Dict[str, Set[Any]]) -> None:
        self.token_lookup = token_lookup or {}
        self.id_map_lookup = id_map_lookup or {}
        self.id_array_lookup = id_array_lookup or {}
        self.token_to_entity_lookup = token_to_entity_lookup or {}

        self.is_tensorized = False
        self.global_id_lookup: List[torch.Tensor] = []
        self.local_id_lookup: List[torch.Tensor] = []
        self.token_id_to_entity_id_lookup: List[torch.Tensor] = []
        self._num_entities = -1

    @classmethod
    def load(cls, path: str):

        logger.info('Loading alias database from "%s". This will probably take a second.', path)
        tokenizer = SpacyTokenizer()
        token_lookup: Dict[str, AliasList] = {}
        id_map_lookup: Dict[str, Dict[str, int]] = {}
        id_array_lookup: Dict[str, np.ndarray] = {}
        token_to_entity_lookup: Dict[str, Set[Any]] = defaultdict(set)

        # Right now we only support loading the alias database from a pickle file.
        with open(path, 'rb') as f:
            alias_lookup = pickle.load(f)

        for entity, aliases in tqdm(alias_lookup.items()):
            # Reverse token to potential entity lookup
            for alias in aliases:
                for token in tokenizer.tokenize(alias):
                    token_to_entity_lookup[token].add(entity)

            # Start by tokenizing the aliases
            tokenized_aliases: AliasList = [tokenizer.tokenize(alias)[:DEFAULTS.max_alias_tokens] for alias in aliases]
            tokenized_aliases = tokenized_aliases[:DEFAULTS.max_alias_num]
            token_lookup[entity] = tokenized_aliases

            # Next obtain the set of unique tokens appearing in aliases for this entity. Use this
            # to build a map from tokens to their unique id.
            unique_tokens = set()
            for tokenized_alias in tokenized_aliases:
                unique_tokens.update(tokenized_alias)
            id_map = {token: i + 1 for i, token in enumerate(unique_tokens)}
            id_map_lookup[entity] = id_map

            # Lastly create an array associating the tokens in the alias to their corresponding ids.
            num_aliases = len(tokenized_aliases)
            max_alias_length = max(len(tokenized_alias) for tokenized_alias in tokenized_aliases)
            id_array = np.zeros((num_aliases, max_alias_length), dtype=int)
            for i, tokenized_alias in enumerate(tokenized_aliases):
                for j, token in enumerate(tokenized_alias):
                    id_array[i, j] = id_map[token]
            id_array_lookup[entity] = id_array

        return cls(token_lookup=token_lookup,
                   id_map_lookup=id_map_lookup,
                   id_array_lookup=id_array_lookup,
                   token_to_entity_lookup=token_to_entity_lookup)

    def token_to_uid(self, entity: str, token: str) -> int:
        if entity in self.id_map_lookup:
            id_map = self.id_map_lookup[entity]
            if token in id_map:
                return id_map[token]
        return 0

    # noinspection PyTypeChecker
    def tensorize(self, raw_ent_vocab: Vocab, tokens_vocab: Vocab, ent_vocab: Vocab):
        """
        Creates a list of tensors from the alias lookup.

        After dataset creation, we'll mainly want to work with alias lists as lists of padded
        tensors and their associated masks. This needs to be done **after** the vocabulary has
        been created. Accordingly, in our current approach, this method must be called in the
        forward pass of the model (since the operation is rather expensive we'll make sure that
        it doesn't anything after the first time it is called).
        """
        # This operation is expensive, only do it once.
        if self.is_tensorized:
            return

        print('Converting AliasDatabase to tensors')

        # entity_idx_to_token = vocab.get_index_to_token_vocabulary('raw_entity_ids')

        for i in range(len(raw_ent_vocab)):  # pylint: disable=C0200
            entity = raw_ent_vocab.id_to_tok[i]
            try:
                tokenized_aliases = self.token_lookup[entity]
            except KeyError:
                # If we encounter non-entity tokens (e.g. padding and null) then just add
                # a blank placeholder - these should not be encountered during training.

                # TODO: raw entity ID has properties (P10 ...) in it. But these don't occur in token lookup
                # is it okay to skip them as well?

                self.global_id_lookup.append(None)
                self.local_id_lookup.append(None)
                continue

            # Construct tensor of alias token indices from the global vocabulary.
            num_aliases = len(tokenized_aliases)
            max_alias_length = max(len(tokenized_alias) for tokenized_alias in tokenized_aliases)
            global_id_tensor = torch.zeros(num_aliases, max_alias_length, dtype=torch.int64,
                                           requires_grad=False)
            for j, tokenized_alias in enumerate(tokenized_aliases):
                for k, token in enumerate(tokenized_alias):
                    # WARNING: Extremely janky cast to string
                    global_id_tensor[j, k] = tokens_vocab.get(str(token), tokens_vocab.unk)
            self.global_id_lookup.append(global_id_tensor)

            # Convert array of local alias token indices into a tensor
            local_id_tensor = torch.tensor(self.id_array_lookup[entity],
                                           requires_grad=False)  # pylint: disable=not-callable
            self.local_id_lookup.append(local_id_tensor)

        # Build the tensorized token -> potential entities lookup.
        # NOTE: Initial approach will be to store just the necessary info to build one-hot vectors
        # on the fly since storing them will probably be way too expensive.
        for token in tokens_vocab.id_to_tok:
            # token = token_idx_to_token[i]
            try:
                potential_entities = self.token_to_entity_lookup[token]
            except KeyError:
                self.token_id_to_entity_id_lookup.append(None)
            else:
                # TODO: figure out what do they do with unknowns? Do unknowns exist here in this tensor?
                # Like there's an literal in the entities. What to do with it? Ignore it, or encode it?
                potential_entity_ids = torch.tensor(
                    ent_vocab.encode(potential_entities, ignore_unknowns=True),
                    dtype=torch.int64,
                    requires_grad=False)
                self.token_id_to_entity_id_lookup.append(potential_entity_ids)
        self._num_entities = len(ent_vocab)  # Needed to get one-hot vector length

        self.is_tensorized = True

        print('Done converting AliasDatabase data to tensors.')

    # noinspection PyArgumentList
    def lookup(self, entity_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Looks up alias tokens for the given entities."""
        # Initialize empty tensors and fill them using the lookup
        batch_size, sequence_length = entity_ids.shape
        global_tensor = entity_ids.new_zeros(batch_size, sequence_length, DEFAULTS.max_alias_num,
                                             DEFAULTS.max_alias_tokens, requires_grad=False)
        local_tensor = entity_ids.new_zeros(batch_size, sequence_length, DEFAULTS.max_alias_num,
                                            DEFAULTS.max_alias_tokens, requires_grad=False)
        for i in range(batch_size):
            for j in range(sequence_length):
                entity_id = entity_ids[i, j]
                local_indices = self.local_id_lookup[entity_id]
                global_indices = self.global_id_lookup[entity_id]
                if local_indices is not None:
                    num_aliases, alias_length = local_indices.shape
                    local_tensor[i, j, :num_aliases, :alias_length] = local_indices
                    global_tensor[i, j, :num_aliases, :alias_length] = global_indices

        return global_tensor, local_tensor

    def reverse_lookup(self, tokens: torch.Tensor) -> torch.Tensor:
        """Looks up potential entity matches for the given token."""
        batch_size, sequence_length = tokens.shape
        logger.debug('Performing reverse lookup')
        # noinspection PyArgumentList
        output = tokens.new_zeros(batch_size, sequence_length, self._num_entities,
                                  dtype=torch.uint8,
                                  requires_grad=False)
        for i in range(batch_size):
            for j in range(sequence_length):
                token_id = tokens[i, j]
                potential_entities = self.token_id_to_entity_id_lookup[token_id]
                output[i, j, potential_entities] = 1
        return output


if __name__ == '__main__':

    # Try to init an instance and see what happens
    ad = AliasDatabase.load(path='data/linked-wikitext-2/alias.pkl')
    print(len(ad.token_lookup))

    print('potato')
