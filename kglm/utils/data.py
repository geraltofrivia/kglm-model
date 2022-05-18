"""
    Define data classes to mimic allennlp instances.
    Make it super specific to the data we've been given. That's fine.
"""
from dataclasses import dataclass, asdict
from typing import List, Dict
import numpy as np


@dataclass
class Instance:
    """
        Needed functionality:
            -> automatically return multiple objects of same class, split based on some criteria
            -> give info on what field does what (metadata vs actual content)

    """
    # TODO rename this dataclass

    source: List[str]                       # x labels for LM task
    target: List[str]                       # y labels for LM task

    entities: List[str]                     # entity ID for every token
    relations: List[str]                    # relation ID for every token
    raw_entities: List[str]                 # same as entities, having entity info or pad tokens
    parent_ids: List[List[str]]             # for every token, have a list of all parent entity (dynamic graph in KGLM)

    shortlist: List[str]                    # list of entities (all entities? je sais pas)
    reverse_shortlist: Dict[str, int]       # similar to shortlist but w index info of all entities there

    # These are all floats
    shortlist_inds: np.ndarray              # TODO: what is it again?
    mention_type: np.ndarray                # TODO: what is it again?
    alias_copy_inds: np.ndarray             # TODO: what is it again?

    def split(self):
        # TODO
        ...

    @property
    def metadata(self):
        # TODO: if there are some metadata fields, add them here down the line
        return []

    @staticmethod
    def empty():
        return Instance(source=[], target=[], entities=[], relations=[], raw_entities=[], parent_ids=[],
                        shortlist=[], reverse_shortlist={},
                        shortlist_inds=np.array([], dtype=np.float64),
                        mention_type=np.array([], dtype=np.float64),
                        alias_copy_inds=np.array([], dtype=np.float64))

    # noinspection PyUnresolvedReferences
    @property
    def fields(self) -> List[str]:
        """ Expose the keys of __dataclass_fields__ """
        return list(self.__dataclass_fields__.keys())

    def asdict(self) -> dict:
        return asdict(self)


if __name__ == '__main__':
    print(Instance.empty())
