""" the bottom tier of random assortment of things. DO NOT IMPORT ANY LOCAL THING HERE.
Avoid cyclic dependencies for a healthier and longer life. """
from typing import Union, List, Tuple, Dict, Type
from pathlib import Path
from mytorch.utils.goodies import FancyDict
import torch
import json


def change_device(instance: dict, device: Union[str, torch.device]) -> dict:
    """ Go through every k, v in a dict and change its device (make it recursive) """
    for k, v in instance.items():
        if type(v) is torch.Tensor:
            # if 'device' == 'cpu':
            instance[k] = v.detach().to(device)
            # else:
            #     instance[k] = v.to(device)
        elif type(v) is dict:
            instance[k] = change_device(v, device)

    return instance


def merge_configs(old, new):
    """
        we copy over elements from old and add them to new IF the element does not exist in new.
            If the element is a dict, we do this recursively.

        arg new may be a dict or a FancyDict or a BertConfig
    """

    if type(new) is dict:
        new = FancyDict(new)
        # new_caller = __getattribute__

    for k, v in old.items():

        try:
            _ = new.__getattr__(k)

            # Check if the Value is nested
            if type(v) in [FancyDict, dict]:
                # If so, call the fn recursively
                v = merge_configs(v, new.__getattr__(k))
                new.__setattr__(k, v)
        except (AttributeError, KeyError) as e:
            new.__setattr__(k, v)

    return new


def pull_embeddings_from_disk(p: Path, tok2id: Dict[str, int]) -> torch.Tensor:
    """
        Similar to AllenNLP's thing, we pull vectors from disk. And align them to the provided tok2id vocab.
        If some elements are not found in the vocab, we randomly init them with a mean and std of the matrix we do have.
    Parameters
    ----------
    p
    tok2id

    Returns
    -------

    """
    if not p.exists():
        raise FileNotFoundError(f"{str(p)} does not exist")

    local_tok2id = {}
    local_vectors = []
    for line in p.open('r').readlines():
        item, vector = line.split()[0], torch.tensor([float(x) for x in line.split()[1:]], dtype=torch.float)
        local_tok2id[item] = len(local_tok2id)
        local_vectors.append(vector)

    local_vectors = torch.stack(local_vectors)
    _mean, _std = torch.mean(local_vectors), torch.std(local_vectors)
    _dim = local_vectors.shape[1]
    vectors = torch.empty(len(tok2id), _dim).normal_(mean=_mean, std=_std)

    # Try to align the vectors to the given vocab; and note the ones which don't fit.
    # unknown_indices = []
    for global_tok, global_index in tok2id.items():
        try:
            local_index = local_tok2id[global_tok]
        except KeyError:
            # unknown_indices.append(global_index)
            continue
        local_vector = local_vectors[local_index]
        vectors[global_index] = local_vector

    # unknown_vectors = torch.empty(len(unknown_indices), _dim).normal_(mean=_mean, std=_std)
    # vectors[unknown_indices] = unknown_vectors

    return vectors


def serialize_config(config: dict):
    """ Safely handle things """
    try:
        # First see if you need to make any change
        _ = json.dumps(config)
        return config
    except TypeError as e:
        serialized = {}
        for k, v in config.items():
            try:
                serialized[k] = json.dumps(v)
            except TypeError:
                if isinstance(v, Path):
                    serialized[k] = json.dumps(str(v))
                elif isinstance(v, torch.Tensor):
                    try:
                        serialized[k] = json.dumps(v.item())
                    except ValueError:
                        # This is not a 1D tensor, its an array. Too bad.
                        serialized[k] = json.dumps(v.cpu().tolist())
                else:
                    raise ValueError(f"Unknown type of object - {type(v)}. If you think its serializable, "
                                     f"handle it in code here manually.")
        return serialized





