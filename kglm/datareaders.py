"""
    Here we keep all data readers
"""
import json
import numpy as np
from typing import Iterable
# from overrides import overrides

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from utils.alias import AliasDatabase
from utils.exceptions import ConfigurationError
from utils.data import Instance
from config import DEFAULT_PAD_TOKEN, MAX_PARENTS, LOCATIONS as LOC


def normalize_entity_id(raw_entity_id: str) -> str:
    if raw_entity_id[0] == 'T':
        entity_id = '@@DATE@@'
    elif raw_entity_id[0] == 'V':
        entity_id = '@@QUANTITY@@'
    elif raw_entity_id[0] in ['P', 'Q']:
        entity_id = raw_entity_id
    else:
        entity_id = None
    return entity_id


class EnhancedWikitextKglmReader:

    def __init__(self,
                 alias_database_path: str,
                 mode: str = "generative",
                 lazy: bool = False) -> None:
        """
        Parameters
        ----------
        alias_database_path : str
            Path to the alias database.
        mode : str, optional (default="generative")
            One of "discriminative" or "generative", indicating whether generated
            instances are suitable for the discriminative or generative version of
            the model.
        """
        # super().__init__(lazy)
        if mode not in {"discriminative", "generative"}:
            raise ConfigurationError("Got mode {}, expected one of 'generative'"
                                     "or 'discriminative'".format(mode))
        self._mode = mode
        self.alias_database = AliasDatabase.load(path=alias_database_path)

    # @overrides
    def load(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as f:
            for line in f:
                data = json.loads(line)

                # Extract tokens
                tokens = data['tokens']
                source = tokens[:-1]
                if self._mode == 'generative':
                    target = tokens[1:]
                else:
                    target = None

                # Process annotations
                if 'annotations' not in data:
                    shortlist = None
                    reverse_shortlist = None
                    raw_entity_ids = None
                    entity_ids = None
                    relations = None
                    parent_ids = None
                    shortlist_inds = None
                    mention_type = None
                else:
                    # We maintain a "shortlist" of observed entities, that is used for baseline models
                    # that only select entities from the set that appear in the document (as opposed to
                    # the set of all possible entities).
                    shortlist = [DEFAULT_PAD_TOKEN]
                    reverse_shortlist = {DEFAULT_PAD_TOKEN: 0}
                    raw_entity_ids = [DEFAULT_PAD_TOKEN] * len(source)
                    entity_ids = [DEFAULT_PAD_TOKEN] * len(source)
                    relations = [[DEFAULT_PAD_TOKEN]] * len(source)
                    parent_ids = [[DEFAULT_PAD_TOKEN]] * len(source)
                    shortlist_inds = np.zeros(shape=(len(source),))
                    mention_type = np.zeros(shape=(len(source),))

                    if self._mode == "generative":
                        alias_copy_inds = np.zeros(shape=(len(source),))
                    else:
                        alias_copy_inds = None

                    # Process annotations
                    for annotation in data['annotations']:

                        # Obtain the entity identifier for the annotated span
                        raw_entity_id = annotation['id']
                        raw_parent_id = annotation['parent_id']
                        entity_id = normalize_entity_id(raw_entity_id)
                        if entity_id is None:
                            continue
                        parent_id = [normalize_entity_id(x) for x in raw_parent_id]
                        assert len(parent_id) == len(raw_parent_id)
                        relation = annotation['relation']
                        new_entity = relation == ['@@NEW@@']

                        # If necessary, update the shortlist. Obtain the index of the entity identifier in
                        # the shortlist.
                        if entity_id not in reverse_shortlist:
                            reverse_shortlist[entity_id] = len(reverse_shortlist)
                            shortlist.append(entity_id)
                        shortlist_ind = reverse_shortlist[entity_id]

                        # Update the outputs
                        # Offset is 0 in generative case, since each timestep is for predicting
                        # attributes of the next token. In the discriminative case, each timestep
                        # is for predicting attributes of the current token.
                        mode_offset = -1 if self._mode == "generative" else 0
                        span = annotation['span']
                        for i in range(*span):
                            raw_entity_ids[i + mode_offset] = raw_entity_id
                            entity_ids[i + mode_offset] = entity_id
                            mention_type[i + mode_offset] = 3
                            if new_entity:
                                shortlist_inds[i + mode_offset] = shortlist_ind
                            else:
                                relations[i + mode_offset] = relation[:MAX_PARENTS]
                                parent_ids[i + mode_offset] = parent_id[:MAX_PARENTS]
                            if self._mode == "generative":
                                alias_copy_inds[i + mode_offset] = self.alias_database.token_to_uid(raw_entity_id, tokens[i])
                                # TODO: alias_copy_inds not the same as original KGLM
                        # Now put in proper mention type for first token
                        start = annotation['span'][0]
                        if new_entity:
                            mention_type[start + mode_offset] = 1
                        else:
                            mention_type[start + mode_offset] = 2

                yield Instance(
                    source=source,
                    target=target,
                    entities=entity_ids,
                    relations=relations,
                    raw_entities=raw_entity_ids,
                    parent_ids=parent_ids,
                    shortlist=shortlist,
                    reverse_shortlist=reverse_shortlist,
                    shortlist_inds=shortlist_inds,
                    mention_type=mention_type,
                    alias_copy_inds=alias_copy_inds
                )


if __name__ == '__main__':

    # Lets try and ge the datareader to work also
    ds = EnhancedWikitextKglmReader(alias_database_path=LOC.lw2 / 'alias.pkl')
    for inst in ds.load(LOC.lw2 / 'train.jsonl'):
        for k in inst.keys():
            print(k)
        print('potato')
        break
