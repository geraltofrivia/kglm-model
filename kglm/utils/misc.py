""" the bottom tier of random assortment of things. DO NOT IMPORT ANY LOCAL THING HERE.
Avoid cyclic dependencies for a healthier and longer life. """
from typing import Union
from mytorch.utils.goodies import FancyDict
import torch


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
