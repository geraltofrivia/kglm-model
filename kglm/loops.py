""" Here be ze training loops """

'''
# TODOs

[ ] Metrics computation
[ ] WandB integration
[ ] Model saving
'''
from tqdm.auto import tqdm
import torch.nn
from typing import Callable, Dict, Optional, Type
from pathlib import Path
import numpy as np

# Local imports
from utils.misc import change_device
from utils.exceptions import FoundNaNs


# noinspection PyUnresolvedReferences
def training_loop(
        model: torch.nn.Module,
        forward_fn: Callable,
        train_dl: Callable,
        valid_dl: Callable,
        device: str,
        epochs: int,
        optim: torch.optim,
        scheduler: Optional[Type[torch.optim.lr_scheduler._LRScheduler]],
        clip_grad_norm: float = 0.0,

        # WandB Logging Stuff
        flag_wandb: bool = False,
        epochs_last_run: int = 0,

        # flag_save: bool = False,
        # save_config: Dict = None,
        # save_dir: Optional[Path] = None,
):

    train_loss = []
    train_metrics: Dict = {}    # TODO: what metrics are we using anyway
    valid_metrics: Dict = {}    # TODO: same as above

    # Epoch Level
    for e in range(epochs_last_run + 1, epochs + epochs_last_run + 1):

        # Make data
        trn_dataset = train_dl()
        vld_dataset = valid_dl

        # Bookkeeping stuff
        per_epoch_loss = []

        # Set the model mode to train
        model.train()

        for i, instance in enumerate(tqdm(trn_dataset)):

            # Reset the gradients
            optim.zero_grad()

            instance = change_device(instance, device)

            outputs = forward_fn(**instance)
            loss = outputs['loss']

            if torch.isnan(loss):
                raise FoundNaNs(f"Found NaN in the loss. Epoch: {e}, Iteration: {i}.")

            loss.backward()

            # Clip Gradients
            if clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [param for group in optim.param_groups for param in group['params']], clip_grad_norm
                )

            # Update parameters
            optim.step()

            # TODO: note down the metrics !
            per_epoch_loss.append(loss.item())

        # TODO: change model mode to eval
        # TODO: evaluation snippet

        # If LR scheduler is provided, run it
        if scheduler is not None:
            scheduler.step()

        # Bookkeeping
        train_loss.append(np.mean(per_epoch_loss))
        if flag_wandb:
            wandb.log({
                'loss': train_loss[-1],
            })

        print(f"\nEpoch: {e:5d}" +
              f"\n\tLoss: {train_loss[-1]:.8f}")

        # Saving

        # DONE



