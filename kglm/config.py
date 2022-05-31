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
    }
)
LOCATIONS.vocab = LOCATIONS.lw2 / 'vocab'

for k, v in LOCATIONS.items():
    if not v.exists():
        raise AssertionError(f"Expected to find '{k}' in {v}. But its not there. Fix it svp! "
                             f"Current dir is {os.getcwd()}. "
                             f"Absolute path is {v.resolve()}")

MAX_PARENTS = 10
# AllenNLP stuff
DEFAULT_PAD_TOKEN = '@@PADDING@@'
DEFAULT_UNK_TOKEN = '@@UNKNOWN@@'
DEFAULT_BOS_TOKEN = '@@START@@'
DEFAULT_EOS_TOKEN = '@@END@@'

# Used in utils/alias.py
MAX_ALIAS_NUM = 4
MAX_ALIAS_TOKENS = 8
