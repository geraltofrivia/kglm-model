"""
    Here we keep all data readers
"""
from typing import Dict

from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from utils.exceptions import ConfigurationError

class EnhancedWikitextKglmReader:

    def __init__(self,
                 alias_database_path: str,
                 mode: str = "generative",
                 token_indexers: Dict[str, TokenIndexer] = None,
                 entity_indexers: Dict[str, TokenIndexer] = None,
                 raw_entity_indexers: Dict[str, TokenIndexer] = None,
                 relation_indexers: Dict[str, TokenIndexer] = None,
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
        super().__init__(lazy)
        if mode not in {"discriminative", "generative"}:
            raise ConfigurationError("Got mode {}, expected one of 'generative'"
                                     "or 'discriminative'".format(mode))
        self._mode = mode

        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._entity_indexers = entity_indexers or {'entity_ids': SingleIdTokenIndexer(namespace='entity_ids')}
        self._raw_entity_indexers = raw_entity_indexers or {'raw_entity_ids': SingleIdTokenIndexer(namespace='raw_entity_ids')}
        self._relation_indexers = relation_indexers or {'relations': SingleIdTokenIndexer(namespace='relations')}
        if 'tokens' not in self._token_indexers or \
                not isinstance(self._token_indexers['tokens'], SingleIdTokenIndexer):
            raise ConfigurationError("EnhancedWikitextReader expects 'token_indexers' to contain "
                                     "a 'single_id' token indexer called 'tokens'.")
        if 'entity_ids' not in self._entity_indexers or \
                not isinstance(self._entity_indexers['entity_ids'], SingleIdTokenIndexer):
            raise ConfigurationError("EnhancedWikitextReader expects 'entity_indexers' to contain "
                                     "a 'single_id' token indexer called 'entity_ids'.")
        if 'raw_entity_ids' not in self._raw_entity_indexers or \
                not isinstance(self._raw_entity_indexers['raw_entity_ids'], SingleIdTokenIndexer):
            raise ConfigurationError("EnhancedWikitextReader expects 'raw_entity_indexers' to contain "
                                     "a 'single_id' token indexer called 'raw_entity_ids'.")
        if 'relations' not in self._relation_indexers or \
                not isinstance(self._relation_indexers['relations'], SingleIdTokenIndexer):
            raise ConfigurationError("EnhancedWikitextReader expects 'relation_indexers' to contain "
                                     "a 'single_id' token indexer called 'relations'.")
        self._alias_database = AliasDatabase.load(path=alias_database_path)

    @overrides
    def _read(self, file_path: str) -> Iterable[Instance]:
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
                    shortlist = [DEFAULT_PADDING_TOKEN]
                    reverse_shortlist = {DEFAULT_PADDING_TOKEN: 0}
                    raw_entity_ids = [DEFAULT_PADDING_TOKEN] * len(source)
                    entity_ids = [DEFAULT_PADDING_TOKEN] * len(source)
                    relations = [[DEFAULT_PADDING_TOKEN]] * len(source)
                    parent_ids = [[DEFAULT_PADDING_TOKEN]] * len(source)
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

                        # If neccessary, update the shortlist. Obtain the index of the entity identifier in
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
                            raw_entity_ids[i+mode_offset] = raw_entity_id
                            entity_ids[i+mode_offset] = entity_id
                            mention_type[i+mode_offset] = 3
                            if new_entity:
                                shortlist_inds[i+mode_offset] = shortlist_ind
                            else:
                                relations[i+mode_offset] = relation[:MAX_PARENTS]
                                parent_ids[i+mode_offset] = parent_id[:MAX_PARENTS]
                            if self._mode == "generative":
                                alias_copy_inds[i+mode_offset] = self._alias_database.token_to_uid(raw_entity_id, tokens[i])
                        # Now put in proper mention type for first token
                        start = annotation['span'][0]
                        if new_entity:
                            mention_type[start+mode_offset] = 1
                        else:
                            mention_type[start+mode_offset] = 2

                yield self.text_to_instance(source,
                                            target,
                                            shortlist,
                                            reverse_shortlist,
                                            raw_entity_ids,
                                            entity_ids,
                                            relations,
                                            parent_ids,
                                            shortlist_inds,
                                            mention_type,
                                            alias_copy_inds)

    @overrides
    def text_to_instance(self,
                         source,
                         target=None,
                         shortlist=None,
                         reverse_shortlist=None,
                         raw_entity_ids=None,
                         entity_ids=None,
                         relations=None,
                         parent_ids=None,
                         shortlist_inds=None,
                         mention_type=None,
                         alias_copy_inds=None) -> Instance:  # pylint: disable=arguments-differ
        metadata = {
            'source_tokens': source,
            'alias_database': self._alias_database
        }
        fields = {
            'metadata': MetadataField(metadata),
            'source': TextField(_tokenize(source), self._token_indexers),
        }

        if target is not None:
            fields['target'] = TextField(_tokenize(target), self._token_indexers)
            metadata['target_tokens'] = target
        if shortlist is not None:
            fields['shortlist'] = TextField(_tokenize(shortlist), self._entity_indexers)
        if raw_entity_ids is not None:
            fields['raw_entity_ids'] = TextField(_tokenize(raw_entity_ids), self._raw_entity_indexers)
        if entity_ids is not None:
            fields['entity_ids'] = TextField(_tokenize(entity_ids), self._entity_indexers)
        if parent_ids is not None:
            fields['parent_ids'] = ListField([
                TextField(_tokenize(sublist),
                          token_indexers=self._entity_indexers)
                for sublist in parent_ids])
        if relations is not None:
            fields['relations'] = ListField([
                TextField(_tokenize(sublist),
                          token_indexers=self._relation_indexers)
                for sublist in relations])
        if mention_type is not None:
            fields['mention_type'] = SequentialArrayField(mention_type, dtype=np.int64)
        if shortlist_inds is not None:
            fields['shortlist_inds'] = SequentialArrayField(shortlist_inds, dtype=np.int64)
        if alias_copy_inds is not None:
            fields['alias_copy_inds'] = SequentialArrayField(alias_copy_inds, dtype=np.int64)

        return Instance(fields)