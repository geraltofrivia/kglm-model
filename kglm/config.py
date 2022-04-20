from mytorch.utils.goodies import FancyDict
from pathlib import Path
from typing import Dict

ROOT_LOC: Path = Path("..") if str(Path().cwd()).split("/")[-1] == "src" else Path(".")
LOCATIONS: Dict[str, Path] = FancyDict(
    **{
        "root": ROOT_LOC,
        "lw2": ROOT_LOC / 'data' / 'linked-wikitext-2'
    }
)

for k, v in LOCATIONS.items():
    if not v.exists():
        raise AssertionError(f"Expected to find '{k}' in {v}. But its not there. Fix it svp!")

MAX_PARENTS = 10
# AllenNLP stuff
DEFAULT_PADDING_TOKEN = "@@PADDING@@"


# Used in utils/alias.py
MAX_ALIAS_NUM = 4
MAX_ALIAS_TOKENS = 8
