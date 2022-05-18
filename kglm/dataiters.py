"""
    Datareaders read from disk and maybe preproc.
    Dataiters batch it and make it training friendly.

    This is 2022. Keep up.
"""
import random
import numpy as np
from tqdm.auto import tqdm
from collections import deque
from typing import List, Tuple, Iterable, Dict, Deque, Iterator

# Local Imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from utils.data import Instance
from config import LOCATIONS as LOC
from utils.exceptions import ConfigurationError
from tokenizer import SpacyTokenizer
from datareaders import EnhancedWikitextKglmReader
from utils.vocab import Vocab


class FancyIterator:
    """
        This is the iterator that takes documents processed by EnhancedWikitextKglmReader
            and makes them ready for training (encapsulated in AllenNLP gunk but we can take care of that)...
    """

    def __init__(self,
                 batch_size: int,
                 split_size: int,
                 tokenizer: SpacyTokenizer,
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

    def _batch_convert_(self, batch: List[Dict]):
        """ wrapper over self.vocab.batch_convert """

        # Go through all text fields in instance and convert them to nice crisp tensors
        relevant_fields = ['source', 'target']

        for field in relevant_fields:

            # create a value column
            values = [instance[field] for instance in batch]

            # Proc it
            processed = self.tokenizer.batch_convert(values, pad=True, to='torch')

            # Put it back
            for i, instance in enumerate(batch):
                instance.__setattr__(field, processed[i])

        return batch

    def __call__(self,
                 instances: Iterable[Instance],
                 num_epochs: int = 1,
                 starting_epoch: int = 0,
                 shuffle: bool = False) -> Iterable[Dict]:

        # In order to ensure that we are (almost) constantly streaming data to the model we
        # need to have all of the instances in memory ($$$)
        instance_list = list(instances)

        if (self._batch_size > len(instance_list)) and self._truncate:
            raise ConfigurationError('FancyIterator will not return any data when the batch size '
                                     'is larger than number of instances and truncation is enabled. '
                                     'To fix this either use a smaller batch size (better for '
                                     'training) or disable truncation (better for validation).')

        epochs = range(starting_epoch, starting_epoch + num_epochs)

        for epoch in epochs:

            if shuffle:
                random.shuffle(instance_list)

            # We create queues for each instance in the batch, and greedily fill them to try and
            # ensure each queue's length is roughly equal in size.
            queues: List[Deque[Dict]] = [deque() for _ in range(self._batch_size)]
            queue_lengths = np.zeros(self._batch_size, dtype=int)
            for instance in tqdm(instance_list):
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
                #     # This changes text fields into vocabbed ints
                #     # batch.index_instances(self.vocab)
                #     batch = self.vocab.batch_tokenize(batch)
                #
                # padding_lengths = batch.get_padding_lengths()
                # # this pads and converts to torch tensors.
                # # we could do both things here if needed
                # yield batch.as_tensor_dict(padding_lengths), 1
                yield self._batch_convert_(batch)

    def _split_instance(self, instance: Dict) -> Tuple[List[Dict], int]:
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
        chunks: List[Dict] = []

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

        return chunks, padded_length

    def _generate_batches(
            self,
            queues: List[Deque[Dict]],
            blank_instance: Instance) -> List[Dict]:
        num_iter = max(len(q) for q in queues)
        for _ in range(num_iter):
            instances: List[Instance] = []
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
        return 0


if __name__ == '__main__':
    # Lets try and ge the datareader to work
    ds = EnhancedWikitextKglmReader(alias_database_path=LOC.lw2 / 'alias.pkl')

    # Pull a vocab
    vocab = Vocab.load(LOC.vocab / 'tokens.txt')

    # Get the vocab and give it to spacy tokenizer.
    tokenizer = SpacyTokenizer(vocab=vocab, pretokenized=True)

    # Now make a dataiter to work with it.
    di = FancyIterator(
        batch_size=10,
        split_size=70,
        tokenizer=tokenizer,
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

    for x in di(ds.load(LOC.lw2 / 'train.jsonl')):
        print(x)
        print('potato')
        break
