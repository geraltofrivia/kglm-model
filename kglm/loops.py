""" Here be ze training loops """

'''

'''
import json
import pickle
from pathlib import Path
from typing import Callable, Dict, Optional, Type

import numpy as np
import torch.nn
import wandb
from tqdm.auto import trange, tqdm

from eval import Evaluator
from utils.exceptions import FoundNaNs
# Local imports
from utils.misc import change_device


def training_loop(
        model: torch.nn.Module,
        forward_fn: Callable,
        train_dl: Callable,
        device: str,
        epochs: int,
        valid_evaluator: Optional[Evaluator],
        optim: torch.optim,
        scheduler: Optional[Type[torch.optim.lr_scheduler._LRScheduler]],
        clip_grad_norm: float = 0.0,

        # Saving Stuff
        flag_save: bool = False,
        save_config: Dict = None,
        save_dir: Optional[Path] = None,
        save_every: int = -1,

        # WandB Logging Stuff
        flag_wandb: bool = False,
        epochs_last_run: int = 0,

        # flag_save: bool = False,
        # save_config: Dict = None,
        # save_dir: Optional[Path] = None,
):

    train_loss = []
    train_metrics: Dict = {}

    # Epoch Level
    for e in trange(epochs_last_run + 1, epochs + epochs_last_run + 1):

        # Make datasets. Fresh for each epoch.
        trn_dataset = train_dl()

        # Bookkeeping stuff
        per_epoch_loss = []

        # Set the model mode to train
        model.train()

        pbar = tqdm(trn_dataset)
        for i, instance in enumerate(pbar):

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

            # calculate metrics and note down loss
            per_epoch_loss.append(loss.item())

        # Evaluating the model.
        model.eval()
        if valid_evaluator:
            valid_evaluator.run()

        # If LR scheduler is provided, run it
        if scheduler is not None:
            scheduler.step()

        # Bookkeeping
        train_loss.append(np.mean(per_epoch_loss))
        train_metrics = Evaluator.aggregate_reports(train_metrics, model.get_metrics(reset=True))
        valid_metrics = valid_evaluator.metrics if valid_evaluator else {}

        if flag_wandb:
            wandb_log = {'loss': train_loss[-1]}
            wandb_log['train']: Evaluator.get_last(train_metrics)
            if valid_evaluator:
                wandb_log['valid']: valid_evaluator.report()
            wandb.log()

        # Saving Code 1 - every epoch
        if flag_save:
            save(save_dir=save_dir, e=e, model=model, opt=optim, scheduler=scheduler, train_loss=train_loss,
                 train_metrics=train_metrics, valid_metrics=valid_metrics, save_config=save_config)

        # Saving Code 2 - every N epochs
        if flag_save and save_every > 0 and e > 0 and e % save_every == 0:
            save(save_dir=save_dir, e=e, model=model, opt=optim, scheduler=scheduler, train_loss=train_loss,
                 train_metrics=train_metrics, valid_metrics=valid_metrics, save_config=save_config, save_suffix=f"_{e}")

        print(f"\nEpoch: {e:5d}" +
              f"\n\tLoss: {train_loss[-1]:.8f}" +
              f"\nTrain Metrics: " +
              '\n'.join(f"\n\t{k}: {v:.8f}" for k, v in Evaluator.get_last(train_metrics).items()) +
              (f"\nValid Metrics: " +
               '\n'.join(f"\n\t{k}: {v:.8f}" for k, v in valid_evaluator.report())) if valid_evaluator else ''
              )


def save(
        save_dir: Path,
        e: int,
        model,
        opt,
        scheduler,
        train_loss,
        train_metrics,
        valid_metrics,
        save_config,
        save_suffix: Optional[str] = '',
):
    """Does the actual saving"""
    # Config
    with (save_dir / f'config{save_suffix}.json').open('w+', encoding='utf8') as f:
        json.dump({**save_config, **{'epochs_last_run': e}}, f)

    # Traces
    with (save_dir / f'traces{save_suffix}.pkl').open('wb+') as f:
        pickle.dump({
            'train_metrics': train_metrics,
            'valid_metrics': valid_metrics,
            'train_loss': train_loss
        }, f)

    # Model
    torch.save({
        'epochs_last_run': e,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': opt.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }, Path(save_dir) / f'torch{save_suffix}.save')
    print(f"Model saved on Epoch {e} at {save_dir}.")
