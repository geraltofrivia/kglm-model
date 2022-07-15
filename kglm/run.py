"""A run file which should get everything sorted, ideally"""

# TODO: these below
'''
    [ ] Implement model saving
    [ ] Implement model resuming
'''

# Global imports
import torch
import wandb
import click
from functools import partial
from mytorch.utils.goodies import FancyDict
from torch.nn import Embedding, LSTM
from typing import Any, Optional, Type

# Local imports
from tokenizer import SpacyTokenizer, SimpleTokenizer
from datareaders import EnhancedWikitextKglmReader
from dataiters import FancyIterator
from config import LOCATIONS as LOC, DEFAULTS, KNOWN_OPTIMIZERS as KNOWN_OC, KNOWN_SCHEDULERS, SCHEDULER_DEFAULTS
from utils.vocab import Vocab
from models.kglm import Kglm
from utils.exceptions import BadParameters
from utils.misc import merge_configs
from loops import training_loop
from eval import PenalizedPerplexity, Perplexity, Evaluator


def make_optimizer(
        model: torch.nn.Module,
        learning_rate: float,
        optimizer_class: str,
        weight_decay: float = 0.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_epsilon: float = 1e-6
) -> torch.optim:
    """
        Make the optimizer based on the optimizer class defined (hardcode torch.optim.Adam for 'adam' etc)

        TODO: shall we optimize the embeddings differently than the rest?

    Parameters
    ----------
    model: the nn.Module
    learning_rate: a float (haven't implemented different LR for different groups)
    optimizer_class: a string signifying which optimizer
    weight_decay: classic, needs no introduction
    adam_beta1: adam specific param
    adam_beta2: adam specific param
    adam_epsilon: adam specific param

    Returns
    -------
    Juste une optimizer
    """

    if optimizer_class == 'adam':
        optimizer_kwargs = {"betas": (adam_beta1, adam_beta2), "eps": adam_epsilon,
                            "lr": learning_rate, "weight_decay": weight_decay}
        optimizer_def = torch.optim.AdamW
    elif optimizer_class == 'sgd':
        optimizer_kwargs = {"lr": learning_rate, "weight_decay": weight_decay}
        optimizer_def = torch.optim.SGD
    else:
        raise BadParameters(f"Unknown optimizer class: {optimizer_class}. We only understand {KNOWN_OC} right now.")

    return optimizer_def(model.parameters(), **optimizer_kwargs)


# noinspection PyProtectedMember
def make_scheduler(opt, lr_schedule: Optional[str]) -> Optional[Type[torch.optim.lr_scheduler._LRScheduler]]:
    if not lr_schedule or lr_schedule in ['constant', 'none']:
        return None

    if lr_schedule == 'gamma':
        lambda1 = lambda epoch: SCHEDULER_DEFAULTS['gamma']['decay_rate'] ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda1)
        return scheduler

    else:
        raise BadParameters(f"Unknown LR Schedule Recipe Name - {lr_schedule}")


@click.command()
@click.option("--epochs", "-e", type=int, default=None, help="Specify the number of epochs for which to train.")
@click.option("--device", "-dv", type=str, default='cpu', help="The device to use: cpu, cuda, cuda:0, ...")
@click.option("--batch-size", "-bs", type=int, default=DEFAULTS.batch_size, help="You know what a batch size is.")
@click.option("--split-size", "-ss", type=int, default=DEFAULTS.split_size,
              help="We split the document in chunks of this number of words.")
@click.option("--learning-rate", "-lr", type=float, default=DEFAULTS.trainer.learning_rate,
              help="Learning rate. Defaults to 3e-4")
@click.option("--optimizer", "-opt", type=click.Choice(KNOWN_OC, case_sensitive=False),
              default=DEFAULTS.trainer.optimizer_class, help="adam or sgd.")
@click.option("--lr-scheduler", "-lrs", default='none', type=click.Choice(KNOWN_SCHEDULERS, case_sensitive=False),
              help="Write 'gamma' to decay the lr. constant or none for nothing.")
@click.option('--use-wandb', '-wb', is_flag=True, default=False,
              help="If True, we report this run to WandB")
@click.option('--wandb-comment', '-wbm', type=str, default=None,
              help="If use-wandb is enabled, whatever comment you write will be included in WandB runs.")
@click.option('--wandb-name', '-wbname', type=str, default=None,
              help="You can specify a short name for the run here as well. ")
def main(
        epochs: int,
        device: str,
        batch_size: int,
        split_size: int,
        learning_rate: float,
        optimizer: str,
        lr_scheduler: str,

        # WandB stuff
        use_wandb: bool = False,
        wandb_comment: str = '',
        wandb_name: str = None,
):
    config: FancyDict = FancyDict()
    config.device = device
    config.batch_size = batch_size
    config.split_size = split_size

    config.trainer = FancyDict()
    config.trainer.learning_rate = learning_rate
    config.trainer.epochs = epochs
    config.trainer.optimizer_class = optimizer
    config.trainer.scheduler_class = lr_scheduler

    config.wandb = use_wandb
    config.wandb_comment = wandb_comment

    config = merge_configs(old=DEFAULTS, new=config)

    # Lets try and ge the datareader to work
    train_data = EnhancedWikitextKglmReader(alias_database_path=LOC.lw2 / 'alias.pkl')
    valid_data = EnhancedWikitextKglmReader(alias_database_path=LOC.lw2 / 'alias.pkl')  # , mode='discriminative')

    # Pull the vocabs
    tokens_vocab = Vocab.load(LOC.vocab / 'tokens.txt')
    ent_vocab = Vocab.load(LOC.vocab / 'entity_ids.txt', skip_bos=True, skip_eos=True)
    rel_vocab = Vocab.load(LOC.vocab / 'relations.txt', skip_bos=True, skip_eos=True)
    raw_ent_vocab = Vocab.load(LOC.vocab / 'raw_entity_ids.txt', skip_bos=True, skip_eos=True)

    # Get the vocab and give it to spacy tokenizer.
    tokenizer = SpacyTokenizer(vocab=tokens_vocab, pretokenized=True)

    # Make other vocabs
    ent_tokenizer = SimpleTokenizer(vocab=ent_vocab)
    rel_tokenizer = SimpleTokenizer(vocab=rel_vocab)
    raw_ent_tokenizer = SimpleTokenizer(vocab=raw_ent_vocab)

    # Now make a dataiter to work with it.
    # Note that these aren't actually functioning datasets. But instead, they are objects which know how to interpret
    # #### raw (or partly preprocessed data) and make it training/eval friendly.
    train_di = FancyIterator(
        batch_size=config.batch_size,
        split_size=config.split_size,
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
    valid_di = FancyIterator(
        batch_size=config.batch_size,
        split_size=config.split_size,
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
        ]
    )
    # We now use the FancyIterator objects' __call__ function and throw actual data to it. But we don't do it just now.
    # We create a partial so`train_data_partial()` will execute `train_di.__call__` with the right params given to it.
    train_data_partial = partial(train_di, train_data.load(LOC.lw2 / 'train.jsonl'),
                                 alias_database=train_data.alias_database)
    valid_data_partial = partial(valid_di, valid_data.load(LOC.lw2 / 'valid.jsonl'),
                                 alias_database=valid_data.alias_database)

    # Make the model
    model_params = {
        "ent_vocab": ent_vocab,
        "rel_vocab": rel_vocab,
        "raw_ent_vocab": raw_ent_vocab,
        "tokens_vocab": tokens_vocab,
        "token_embedder": Embedding(len(tokens_vocab), 400),
        "entity_embedder": Embedding(len(ent_vocab), 256),
        "relation_embedder": Embedding(len(rel_vocab), 256),
        "alias_encoder": LSTM(input_size=400, hidden_size=400, num_layers=3),
        "knowledge_graph_path": str(LOC.lw2 / "knowledge_graph.pkl"),
        "use_shortlist": False,
        "hidden_size": 1150,
        "num_layers": 3,
    }
    # text = "The colleague sitting next to me is [MASK]"

    # Initialize KGLM
    model = Kglm(**model_params)
    print("Model params: ", sum([param.nelement() for param in model.parameters()]))

    # Make optimizer
    opt = make_optimizer(
        model,
        optimizer_class=config.trainer.optimizer_class,
        learning_rate=config.trainer.learning_rate,
        weight_decay=config.trainer.weight_decay,
        adam_beta1=config.trainer.adam_beta1,
        adam_beta2=config.trainer.adam_beta2,
        adam_epsilon=config.trainer.adam_epsilon
    )
    scheduler: Optional[Any] = make_scheduler(opt, lr_schedule=config.trainer.scheduler_class)

    # TODO: this
    # Init the metrics
    metric_classes = [Perplexity, PenalizedPerplexity]
    train_eval = Evaluator(metric_classes=metric_classes)
    valid_eval = Evaluator(
        metric_classes=metric_classes, predict_fn=model.forward, data_loader_callable=valid_data_partial
    )

    # WandB Initialisation
    if use_wandb:
        if 'wandbid' not in config.to_dict():
            config.wandbid = wandb.util.generate_id()

        wandb.init(project="crud-lm", entity="magnet", notes=wandb_comment, name=wandb_name,
                   id=config.wandbid, resume="allow")
        wandb.config.update(config, allow_val_change=True)

    # Create vars and make a training loop
    training_loop(
        model=model,
        forward_fn=model,
        train_dl=train_data_partial,
        train_evaluator=train_eval,
        valid_evaluator=valid_eval,
        device=config.device,
        epochs=config.trainer.epochs,
        optim=opt,
        scheduler=scheduler,
        clip_grad_norm=config.trainer.clip_gradients_norm,
        flag_wandb=use_wandb,
        epochs_last_run=0  # TODO: change if resume is someday implemented
    )

    # See if the dataset works
    # for x in di(ds.load(LOC.lw2 / 'train.jsonl'), alias_database=ds.alias_database):
    # print(x)
    # # print('potato')
    # outputs = model(**x)
    # print('potato')


if __name__ == '__main__':
    main()
