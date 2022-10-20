from collections import Counter
import gc
import logging
from typing import Any, Iterable, Iterator, Tuple, List, Dict
from collections import defaultdict

import torch

# local imports
from utils.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


def log_torch_garbage(verbose=False):
    """Outputs a list / summary of all tensors to the console."""
    logger.debug('Logging PyTorch garbage')
    obj_counts = Counter()
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if verbose:
                    logger.debug('type: %s, size: %s, is_cuda: %s',
                                 type(obj), obj.size(), obj.is_cuda)
                obj_counts[(type(obj), obj.is_cuda)] += 1
            elif hasattr(obj, 'data'):
                if torch.is_tensor(obj.data):
                    if verbose:
                        logger.debug('type: %s, size: %s, is_cuda: %s',
                                     type(obj), obj.size(), obj.is_cuda)
                    obj_counts[(type(obj), obj.is_cuda)] += 1
        except (KeyError, OSError, RuntimeError):
            continue
    logger.debug('Summary stats')
    for key, count in obj_counts.most_common():
        obj_type, is_cuda = key
        logger.debug('type: %s, is_cuda: %s, count: %i', obj_type, is_cuda, count)

def parallel_sample(probs: torch.FloatTensor) -> torch.LongTensor:
    *output_shape, n_categories = probs.shape
    samples = torch.multinomial(probs.view(-1, n_categories), num_samples=1, replacement=True)
    return samples.view(*output_shape)

def sample_from_logp(logp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Draws samples from a tensor of _log probabilities. API matches ``torch.max()``.

    Parameters
    ----------
    logp : ``torch.Tensor``
        Tensor of shape ``(batch_size, ..., n_categories)`` of _log probabilities.

    Returns
    -------
    A tuple consisting of:
    selected_logp : ``torch.Tensor``
        Tensor of shape ``(batch_size, ...)`` containing the selected _log probabilities.
    selected_idx : ``torch.Tensor``
        Tensor of shape ``(batch_size, ...)`` containing the selected indices.
    """
    pdf = torch.exp(logp)
    cdf = torch.cumsum(pdf, dim=-1)
    rng = torch.rand(logp.shape[:-1], device=logp.device).unsqueeze(-1)
    selected_idx = cdf.lt(rng).sum(dim=-1)
    hack = torch.ones(logp.shape[:-1], device=logp.device, dtype=torch.uint8)
    selected_logp = logp[hack, selected_idx[hack]]
    return selected_logp, selected_idx


def nested_enumerate(iterable):
    try:
        for i, element in enumerate(iterable):
            for item in nested_enumerate(element):
                combo = i, *item
                yield combo
    except TypeError:
        yield (iterable,)



def batch_tensor_dicts(tensor_dicts: List[Dict[str, torch.Tensor]],
                       remove_trailing_dimension: bool = False) -> Dict[str, torch.Tensor]:
    """
    Copied from AllenNLP > nn > utils.py

    Takes a list of tensor dictionaries, where each dictionary is assumed to have matching keys,
    and returns a single dictionary with all tensors with the same key batched together.

    Parameters
    ----------
    tensor_dicts : ``List[Dict[str, torch.Tensor]]``
        The list of tensor dictionaries to batch.
    remove_trailing_dimension : ``bool``
        If ``True``, we will check for a trailing dimension of size 1 on the tensors that are being
        batched, and remove it if we find it.
    """
    key_to_tensors: Dict[str, List[torch.Tensor]] = defaultdict(list)
    for tensor_dict in tensor_dicts:
        for key, tensor in tensor_dict.items():
            key_to_tensors[key].append(tensor)
    batched_tensors = {}
    for key, tensor_list in key_to_tensors.items():
        batched_tensor = torch.stack(tensor_list)
        if remove_trailing_dimension and all(tensor.size(-1) == 1 for tensor in tensor_list):
            batched_tensor = batched_tensor.squeeze(-1)
        batched_tensors[key] = batched_tensor
    return batched_tensors


def sort_batch_by_length(tensor: torch.Tensor, sequence_lengths: torch.Tensor):
    """
    Copied from AllenNLP > nn > utils.py

    Sort a batch first tensor by some specified lengths.

    Parameters
    ----------
    tensor : torch.FloatTensor, required.
        A batch first Pytorch tensor.
    sequence_lengths : torch.LongTensor, required.
        A tensor representing the lengths of some dimension of the tensor which
        we want to sort by.

    Returns
    -------
    sorted_tensor : torch.FloatTensor
        The original tensor sorted along the batch dimension with respect to sequence_lengths.
    sorted_sequence_lengths : torch.LongTensor
        The original sequence_lengths sorted by decreasing size.
    restoration_indices : torch.LongTensor
        Indices into the sorted_tensor such that
        ``sorted_tensor.index_select(0, restoration_indices) == original_tensor``
    permuation_index : torch.LongTensor
        The indices used to sort the tensor. This is useful if you want to sort many
        tensors using the same ordering.
    """

    if not isinstance(tensor, torch.Tensor) or not isinstance(sequence_lengths, torch.Tensor):
        raise ConfigurationError("Both the tensor and sequence lengths must be torch.Tensors.")

    sorted_sequence_lengths, permutation_index = sequence_lengths.sort(0, descending=True)
    sorted_tensor = tensor.index_select(0, permutation_index)

    index_range = torch.arange(0, len(sequence_lengths), device=sequence_lengths.device)
    # This is the equivalent of zipping with index, sorting by the original
    # sequence lengths and returning the now sorted indices.
    _, reverse_mapping = permutation_index.sort(0, descending=False)
    restoration_indices = index_range.index_select(0, reverse_mapping)
    return sorted_tensor, sorted_sequence_lengths, restoration_indices, permutation_index