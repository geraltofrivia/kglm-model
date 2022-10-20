import logging
import pickle
from typing import List, Tuple

from tqdm import tqdm
import torch

# Local imports
try:
    import _pathfix
except ImportError:
    from . import _pathfix
from utils.vocab import Vocab
from utils.nn.util import nested_enumerate

logger = logging.getLogger(__name__)


class KnowledgeGraphLookup:
    def __init__(self,
                 knowledge_graph_path: str,
                 ent_vocab: Vocab,
                 rel_vocab: Vocab,
                 raw_ent_vocab: Vocab) -> None:
        self._knowledge_graph_path = knowledge_graph_path
        self._ent_vocab = ent_vocab
        self._rel_vocab = rel_vocab
        self._raw_ent_vocab = raw_ent_vocab
        self._relations, self._tail_ids = self.load_edges(knowledge_graph_path)

    def load_edges(self, knowledge_graph_path: str) -> Tuple[List[torch.LongTensor], List[torch.LongTensor]]:
        logger.info('Loading knowledge graph from: %s', knowledge_graph_path)
        with open(knowledge_graph_path, 'rb') as f:
            knowledge_graph = pickle.load(f)

        entity_idx_to_token = self._ent_vocab.id_to_tok
        all_relations: List[torch.Tensor] = []
        all_tail_ids: List[torch.Tensor] = []
        for i in tqdm(range(len(entity_idx_to_token)),
                      desc=f"Loading knowledge graph from: {knowledge_graph_path}.", position=0, leave=True):
            entity_id = entity_idx_to_token[i]
            try:
                edges = knowledge_graph[entity_id]
            except KeyError:
                relations = None
                tail_ids = None
            else:
                if not edges:
                    relations = None
                    tail_ids = None
                else:
                    # Get the relation and tail id tokens
                    relation_tokens, tail_id_tokens = zip(*knowledge_graph[entity_id])
                    # Index tokens
                    relations = self._rel_vocab.encode(relation_tokens, return_type='torch')
                    tail_ids = self._raw_ent_vocab.encode(tail_id_tokens, return_type='torch')
                    # Convert to tensors
                    relations = torch.LongTensor(relations)
                    tail_ids = torch.LongTensor(tail_ids)
            all_relations.append(relations)
            all_tail_ids.append(tail_ids)

        # TODO: you need tensors here !?
        return all_relations, all_tail_ids

    def __call__(self,
                 parent_ids: torch.LongTensor) -> torch.Tensor:
        """Returns all relations of the form:

            (parent_id, relation, tail_id)

        from the knowledge graph.

        Parameters
        -----------
        parent_ids : ``torch.LongTensor``
            A tensor of arbitrary shape `(N, *)` where `*` means any number of additional
            dimensions. Each element is an id in the knowledge graph.

        Returns
        -------
        A tuple `(relations, tail_ids)` containing the following elements:
        relations : ``torch.LongTensor``
            A tensor of shape `(N, *, K)` corresponding to the input shape, with an
            additional dimension whose size `K` is the largest number of relations to
            a parent in the input. Each element is a relation id.
        tail_ids : ``torch.LongTensor``
            A tensor of shape `(N, *, K)` containing the corresponding tail ids.
        """

        # Collect the information to load into the output tensors.
        indices: List[Tuple[int, ...]] = []
        parent_ids_list: List[torch.LongTensor] = []
        relations_list: List[torch.LongTensor] = []
        tail_ids_list: List[torch.LongTensor] = []

        # i = 0
        # j = 0
        for *inds, parent_id in nested_enumerate(parent_ids):
            # Retrieve data
            # i += 1
            relations = self._relations[parent_id]
            tail_ids = self._tail_ids[parent_id]
            if relations is None:
                # j += 1
                continue
            # Add to lists
            indices.append(tuple(inds))
            parent_ids_list.append(parent_id)
            relations_list.append(relations.to(device=parent_ids.device))
            tail_ids_list.append(tail_ids.to(device=parent_ids.device))

        # print("loop is run ", i, " times")      # 42000
        # print("times skipped ", j)
        # print("parent IDs ", parent_ids.shape)  # 60, 70, 10
        # print("indices len", len(indices))      #
        # # print(indices[0])
        # print("parent_ids_list: ", len(parent_ids_list), parent_ids_list[0].shape)
        # # print(parent_ids_list[0])
        # print("relations: ", len(relations_list), relations_list[0].shape)
        # # print(relations_list[0])
        # print("tail_ids_list", len(tail_ids_list), tail_ids_list[0].shape)
        # # print(tail_ids_list[0])

        return indices, parent_ids_list, relations_list, tail_ids_list
