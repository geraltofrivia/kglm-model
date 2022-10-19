"""
    Data readers read from disk and maybe preproc.
    Dataiters batch it and make it training friendly.

    This is 2022. Keep up.
"""
import torch
import random
import numpy as np
from tqdm.auto import tqdm
from collections import deque
from dataclasses import fields
from typing import List, Tuple, Iterable, Dict, Deque, Optional

# Local Imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from utils.data import Instance
from config import LOCATIONS as LOC
from utils.exceptions import ConfigurationError
from tokenizer import SpacyTokenizer, SimpleTokenizer
from datareaders import EnhancedWikitextKglmReader
from utils.vocab import Vocab
from utils.alias import AliasDatabase


class FancyIterator:
    """
        This is the iterator that takes documents processed by EnhancedWikitextKglmReader
            and makes them ready for training (encapsulated in AllenNLP gunk but we can take care of that)...

        Once done with an epoch, just a make a new one, and delete this iter.
    """

    # noinspection PyUnusedLocal
    def __init__(self,
                 batch_size: int,
                 split_size: int,
                 tokenizer: SpacyTokenizer,
                 raw_ent_tokenizer: SimpleTokenizer,
                 rel_tokenizer: SimpleTokenizer,
                 ent_tokenizer: SimpleTokenizer,
                 splitting_keys: List[str],
                 truncate: bool = True,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None,
                 cache_instances: bool = False,
                 track_epoch: bool = False,
                 maximum_samples_per_batch: Tuple[str, int] = None) -> None:
        """
        Parameters
        ----------
        batch_size
        split_size
        splitting_keys: list of keys which are split when batching data up.
            some parts of the data struct should not be split i guess :P
        truncate
        instances_per_epoch
        max_instances_in_memory
        cache_instances
        track_epoch
        maximum_samples_per_batch
        """

        self._splitting_keys = splitting_keys
        self._split_size = split_size  # Set as 70 in kglm.jsonnet
        self._truncate = truncate
        self._batch_size = batch_size

        self.tokenizer = tokenizer
        self.rel_tokenizer = rel_tokenizer
        self.ent_tokenizer = ent_tokenizer
        self.raw_ent_tokenizer = raw_ent_tokenizer

    def _batch_convert_(self, batch: List[Instance], alias_database: AliasDatabase):
        """
            A lot of interesting things happen here.
            We convert elements to tensors, for one.
            Then we fit the data to exactly match what is required in model forward. That is,

                source: Dict[str, torch.Tensor],
                    it has `words`. it might also have `mask`
                target: Dict[str, torch.Tensor] = None,
                    it has `words`. it might also have `mask`
                reset: torch.Tensor,
                metadata: List[Dict[str, Any]],
                    Needs to be like
                        metadata[0]['alias_database']
                    so every data point needs a alias database field as well. oh well :shrug:
                mention_type: torch.Tensor = None,
                raw_entity_ids: Dict[str, torch.Tensor] = None,
                entity_ids: Dict[str, torch.Tensor] = None,
                parent_ids: Dict[str, torch.Tensor] = None,
                relations: Dict[str, torch.Tensor] = None,
                shortlist: Dict[str, torch.Tensor] = None,
                shortlist_inds: torch.Tensor = None,
                alias_copy_inds: torch.Tensor = None,

            NOTE: this assumes that every element in one batch has similar properties.
            That is, if one of them doesnt have a target field, none of them will have a target field
        """

        bs = len(batch)
        outputs = {
            'metadata': [{'alias_database': alias_database} for _ in range(bs)],
            'source': None,
            'target': None,
            'raw_entity_ids': None,
            'entity_ids': None,
            'relations': None,
            'parent_ids': None,
            'shortlist': None,
            'shortlist_inds': None,
            'reset': None,
            'mention_type': None,
            'alias_copy_inds': None
        }

        # Fix source
        if batch[0].source is not None:
            source_values = [instance.source for instance in batch]
            source_words: torch.Tensor = self.tokenizer.batch_convert(source_values, pad=True, to='torch')
            outputs['source']: Optional[Dict[str, torch.Tensor]] = {'words': source_words}

        # Fix target
        if batch[0].target is not None:
            target_values = [instance.target for instance in batch]
            target_words: torch.Tensor = self.tokenizer.batch_convert(target_values, pad=True, to='torch')
            outputs['target']: Optional[Dict[str, torch.Tensor]] = {'words': target_words}

        # Fix raw entity IDs [60, 70]
        if batch[0].raw_entities is not None:
            raw_entity_values = [instance.raw_entities for instance in batch]
            raw_entity_words = self.raw_ent_tokenizer.batch_convert(raw_entity_values, pad=True, to='torch')
            outputs['raw_entity_ids']: Dict[str, torch.Tensor] = {'entity_ids': raw_entity_words}

        # Fix entity IDs [60, 70]
        if batch[0].entities is not None:
            entity_values = [instance.entities for instance in batch]
            entity_words = self.ent_tokenizer.batch_convert(entity_values, pad=True, to='torch')
            outputs['entity_ids']: Dict[str, torch.Tensor] = {'entity_ids': entity_words}

        # Fix Relations. [60, 70, 10]
        if batch[0].relations is not None:
            relations_values = [instance.relations for instance in batch]
            relations_words = self.rel_tokenizer.batch_convert_3d(relations_values, pad=True, to='torch')
            outputs['relations']: Dict[str, torch.Tensor] = {'relations': relations_words}

        # Fix Parents. [60, 70, 10]
        if batch[0].parent_ids is not None:
            parents_values = [instance.parent_ids for instance in batch]
            parents_words = self.ent_tokenizer.batch_convert_3d(parents_values, pad=True, to='torch')
            outputs['parent_ids']: Dict[str, torch.Tensor] = {'entity_ids': parents_words}

        # Fix shortlists (like entities)
        if batch[0].shortlist is not None:
            shortlist_values = [instance.shortlist for instance in batch]
            shortlist_words = self.ent_tokenizer.batch_convert(shortlist_values, pad=True, to='torch')
            outputs['shortlist']: Dict[str, torch.Tensor] = {'entity_ids': shortlist_words}

        # Fix shortlist indices
        if batch[0].shortlist_inds is not None:
            shortlist_ind_values = np.array([instance.shortlist_inds for instance in batch])
            outputs['shortlist_inds']: torch.Tensor = torch.tensor(shortlist_ind_values)

        # Fix reset
        if batch[0].reset is not None:
            reset_values = np.array([instance.reset for instance in batch])
            outputs['reset']: torch.ByteTensor = torch.tensor(reset_values, dtype=torch.uint8)

        # Fix mention types
        if batch[0].mention_type is not None:
            mention_type_values = np.array([instance.mention_type for instance in batch])
            outputs['mention_type']: torch.Tensor = torch.tensor(mention_type_values)

        # Fix shortlist indices
        if batch[0].alias_copy_inds is not None:
            alias_copy_ind_values = np.array([instance.alias_copy_inds for instance in batch])
            outputs['alias_copy_inds']: torch.Tensor = torch.tensor(alias_copy_ind_values)

        # # Go through all text fields in instance and convert them to nice crisp tensors
        # relevant_fields = ['source', 'target']
        #
        # for field in relevant_fields:
        #
        #     # create a value column
        #     values = [instance.__getattribute__(field) for instance in batch]
        #
        #     # Proc it
        #     processed = self.tokenizer.batch_convert(values, pad=True, to='torch')
        #
        #     # Put it back
        #     for i, instance in enumerate(batch):
        #         instance.__setattr__(field, processed[i])

        # TODO: take care of devices etc

        return outputs

    def __call__(self,
                 instances: Iterable[Instance],
                 alias_database: AliasDatabase,
                 num_epochs: int = 1,
                 starting_epoch: int = 0,
                 shuffle: bool = False) -> Iterable[Dict]:
        """

        Parameters
        ----------
        instances: something like
            EnhancedWikitextKglmReader(alias_database_path=LOC.lw2 / 'alias.pkl').load(LOC.lw2 / 'train.jsonl')
            or alternatively, just a list of Instance object
        alias_database: a reference to the alias database used above (or just a relevant alias database).
            not used, only passed to model forward as a part of the batch
        num_epochs
        starting_epoch
        shuffle

        Returns
        -------

        """

        # In order to ensure that we are (almost) constantly streaming data to the model we
        # need to have all of the instances in memory ($$$)
        instance_list = list(instances)

        if (self._batch_size > len(instance_list)) and self._truncate:
            raise ConfigurationError('FancyIterator will not return any data when the batch size '
                                     'is larger than number of instances and truncation is enabled. '
                                     'To fix this either use a smaller batch size (better for '
                                     'training) or disable truncation (better for validation).')

        epochs = range(starting_epoch, starting_epoch + num_epochs)

        for _ in epochs:

            if shuffle:
                random.shuffle(instance_list)

            # We create queues for each instance in the batch, and greedily fill them to try and
            # ensure each queue's length is roughly equal in size.
            queues: List[Deque[Dict]] = [deque() for _ in range(self._batch_size)]
            queue_lengths = np.zeros(self._batch_size, dtype=int)

            # # TODO: REMOVE THIS SUPER URGENTLY!!!!!
            # instance_list = instance_list[:50]

            for instance in tqdm(instance_list,
                                 desc=f"Splitting {len(instance_list)} instances into chunks before batching",
                                 position=0, leave=True):
                # Now we split the instance into chunks.
                chunks, length = self._split_instance(instance.asdict())

                # Next we identify which queue is the shortest and add the chunks to that queue.
                destination = np.argmin(queue_lengths)
                queues[destination].extend(chunks)
                queue_lengths[destination] += length

            # We need a NULL instance to replace the output of an exhausted queue if we are evaluating
            # prototype = deepcopy(chunks[-1])
            # new_fields: dict = {}
            # for name, field in prototype.asdict().items():
            #     if name in prototype.metadata:
            #         new_fields[name] = field
            #     else:
            #         new_fields[name] = []
            blank_instance = Instance.empty()

            for batch in self._generate_batches(queues, blank_instance):
                # if self.vocab is not None:
                #     # This changes text fields into vocabularized ints
                #     # batch.index_instances(self.vocab)
                #     batch = self.vocab.batch_tokenize(batch)
                #
                # padding_lengths = batch.get_padding_lengths()
                # # this pads and converts to torch tensors.
                # # we could do both things here if needed
                # yield batch.as_tensor_dict(padding_lengths), 1
                yield self._batch_convert_(batch, alias_database=alias_database)

    def _split_instance(self, instance: Dict) -> Tuple[List[Instance], int]:
        """Splits one document into multiple batches. That's it. That's all that's happening here."""

        # Determine the size of the sequence inside the instance.
        true_length = len(instance['source'])
        if (true_length % self._split_size) != 0:
            offset = 1
        else:
            offset = 0
        padded_length = self._split_size * (true_length // self._split_size + offset)

        # Determine the split indices.
        split_indices = list(range(0, true_length, self._split_size))
        if true_length > split_indices[-1]:
            split_indices.append(true_length)

        # Determine which fields are not going to be split
        constant_fields = [x for x in instance.keys() if x not in self._splitting_keys]

        # Create the list of chunks
        chunks: List = []

        for i, (start, end) in enumerate(zip(split_indices[:-1], split_indices[1:])):

            # Copy all of the constant fields from the instance to the chunk.
            chunk_fields = {key: instance[key] for key in constant_fields}

            # Determine whether or not to signal model to reset.
            if i == 0:
                reset = np.array(1)
            else:
                reset = np.array(0)
            chunk_fields['reset'] = reset

            # Obtain splits derived from sequence fields.
            for key in self._splitting_keys:
                source_field = instance[key]
                split_field = source_field[start:end]
                chunk_fields[key] = split_field
            chunks.append(chunk_fields)

        # Time to convert these chunks back to instances

        relevant_fields = [field.name for field in fields(Instance)]
        for i, chunk in enumerate(chunks):
            inst = Instance(
                source=chunk['source'],
                target=chunk['target'],
                entities=chunk['entities'],
                relations=chunk['relations'],
                raw_entities=chunk['raw_entities'],
                parent_ids=chunk['parent_ids'],
                shortlist=chunk['shortlist'],
                reverse_shortlist=chunk['reverse_shortlist'],
                shortlist_inds=chunk['shortlist_inds'],
                mention_type=chunk['mention_type'],
                alias_copy_inds=chunk['alias_copy_inds']
            )
            inst.reset = chunk['reset']
            chunks[i] = inst

        return chunks, padded_length

    def _generate_batches(
            self,
            queues: List[Deque[Dict]],
            blank_instance: Instance) -> List[List[Instance]]:
        num_iter = max(len(q) for q in queues)
        for _ in range(num_iter):
            instances: List[Instance] = []

            """
                For every queue, take one (leftmost) instance at a time and make a batch out of it.
                For the first example:
                    there are 10 queues
                    initially each has [3114, 2996, 3007, 2998, 3014, 3078, 3017, 3033, 2994, 2992] instances
                    we run an interator 3114 times and each time take one instance out of each queue 
                        (when queues are depleted, we sub in the blank instance (IF WE ARE EVALUATING))
                        (IF WE ARE TRAINING, we stop as soon as the first queue is dead)                     
            """
            #
            #
            for q in queues:
                try:
                    instance = q.popleft()
                except IndexError:  # A queue is depleted
                    # If we're training, we break to avoid densely padded inputs (since this biases
                    # the model to overfit the longer sequences).
                    if self._truncate:
                        return
                    # But if we're evaluating we do want the padding, so that we don't skip anything.
                    else:
                        instance = blank_instance
                instances.append(instance)
            batch = instances
            yield batch

    @staticmethod
    def get_num_batches(instances: Iterable[Instance]) -> float:
        raise NotImplementedError
        # return 0


if __name__ == '__main__':
    # Lets try and ge the datareader to work
    ds = EnhancedWikitextKglmReader(alias_database_path=LOC.lw2 / 'alias.pkl')

    # Pull the vocabs
    tokens_vocab = Vocab.load(LOC.vocab / 'tokens.txt')
    ent_vocab = Vocab.load(LOC.vocab / 'entity_ids.txt')
    rel_vocab = Vocab.load(LOC.vocab / 'relations.txt')
    raw_ent_vocab = Vocab.load(LOC.vocab / 'raw_entity_ids.txt')

    # Get the vocab and give it to spacy tokenizer.
    tokenizer = SpacyTokenizer(vocab=tokens_vocab, pretokenized=True)

    # Make other vocabs
    ent_tokenizer = SimpleTokenizer(vocab=ent_vocab)
    rel_tokenizer = SimpleTokenizer(vocab=rel_vocab)
    raw_ent_tokenizer = SimpleTokenizer(vocab=raw_ent_vocab)

    # Now make a dataiter to work with it.
    di = FancyIterator(
        batch_size=10,
        split_size=70,
        tokenizer=tokenizer,
        rel_tokenizer=rel_tokenizer,
        ent_tokenizer=ent_tokenizer,
        raw_ent_tokenizer=raw_ent_tokenizer,
        splitting_keys=[
            "source",
            "target",
            "mention_type",
            "raw_entities",
            "entities",
            "parent_ids",
            "relations",
            "shortlist_inds",
            "alias_copy_inds"
        ])

    for x in di(ds.load(LOC.lw2 / 'train.jsonl'), alias_database=ds.alias_database):
        print(x)
        print('potato')
        break
