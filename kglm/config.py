from mytorch.utils.goodies import FancyDict
from pathlib import Path
from typing import Dict

# Local Imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
import os

ROOT_LOC: Path = Path("..") if str(Path().cwd()).split("/")[-1] == "kglm" else Path(".")
LOCATIONS: Dict[str, Path] = FancyDict(
    **{
        "root": ROOT_LOC,
        "lw2": ROOT_LOC / 'data' / 'linked-wikitext-2',
        "models": ROOT_LOC / 'models',
    }
)
LOCATIONS.vocab = LOCATIONS.lw2 / 'vocab'

# for k, v in LOCATIONS.items():
#     if not v.exists():
#         raise AssertionError(f"Expected to find '{k}' in {v}. But its not there. Fix it svp! "
#                              f"Current dir is {os.getcwd()}. "
#                              f"Absolute path is {v.resolve()}")

MAX_PARENTS = 10
# AllenNLP stuff
DEFAULT_PAD_TOKEN = '@@PADDING@@'
DEFAULT_UNK_TOKEN = '@@UNKNOWN@@'
DEFAULT_BOS_TOKEN = '@@START@@'
DEFAULT_EOS_TOKEN = '@@END@@'

KNOWN_OPTIMIZERS = ['adam', 'sgd']
KNOWN_SCHEDULERS = ['gamma', 'constant', 'none']

EMBEDDING_DIM = FancyDict(**{
    'tokens': 400,
    'entities': 256,
    'relations': 256
})

DEFAULTS = FancyDict(**{
    'batch_size':  60,
    'split_size': 70,
    'max_alias_num': 4,
    'max_alias_tokens': 8,
    'tie_weights': True,

    'trainer': FancyDict(**{
        'optimizer_class': 'adam',
        'scheduler_class': 'constant',
        'adam_beta1': 0.9,
        'adam_beta2': 0.999,
        'adam_epsilon': 1e-6,
        'clip_gradients_norm': 0.0,     # TODO: maybe you need to actually clip grads
        'learning_rate': 3e-4,
        'weight_decay': 1.2e-6,
        'scheduler': FancyDict(**{
            'gamma': {'decay_rate': 0.9},
            'none': {},
            'constant': {}
        }),
    }),
    'alias_encoder': FancyDict(**{
        'input_size': EMBEDDING_DIM.tokens,
        'hidden_size': EMBEDDING_DIM.tokens,
        'num_layers': 3,
        'bidirectional': False,
    }),
})

