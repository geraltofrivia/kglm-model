"""A run file which should get everything sorted, ideally"""

# Global imports
from typing import Dict, List,Any
from torch.nn import Embedding, LSTM
import torch

# Local imports
from tokenizer import SpacyTokenizer, SimpleTokenizer
from datareaders import EnhancedWikitextKglmReader
from dataiters import FancyIterator
from config import LOCATIONS as LOC
from utils.vocab import Vocab
from models.kglm import Kglm
from utils.alias import AliasDatabase


def main():

    # Lets try and ge the datareader to work
    ds = EnhancedWikitextKglmReader(alias_database_path=LOC.lw2 / 'alias.pkl')

    # Pull the vocabs
    tokens_vocab = Vocab.load(LOC.vocab / 'tokens.txt')
    ent_vocab = Vocab.load(LOC.vocab / 'entity_ids.txt')
    rel_vocab = Vocab.load(LOC.vocab / 'relations.txt')
    raw_ent_vocab = Vocab.load(LOC.vocab / 'raw_entity_ids.txt')

    # Get the vocab and give it to spacy tokenizer.
    tokenizer = SpacyTokenizer(vocab=tokens_vocab, pretokenized=True)

    # Make other vocabs
    ent_tokenizer = SimpleTokenizer(vocab=ent_vocab)
    rel_tokenizer = SimpleTokenizer(vocab=rel_vocab)
    raw_ent_tokenizer = SimpleTokenizer(vocab=raw_ent_vocab)

    # Now make a dataiter to work with it.
    di = FancyIterator(
        batch_size=10,
        split_size=70,
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

    # Make the model
    MODEL_PARAMS = {
        "ent_vocab": ent_vocab,
        "rel_vocab": rel_vocab,
        "raw_ent_vocab": raw_ent_tokenizer,
        "tokens_vocab": tokens_vocab,
        "token_embedder": Embedding(4, 400),
        "entity_embedder": Embedding(4, 256),
        "relation_embedder": Embedding(4, 256),
        "alias_encoder": LSTM(input_size=400, hidden_size=400, num_layers=3),
        "knowledge_graph_path": "data/linked-wikitext-2/knowledge_graph.pkl",
        "use_shortlist": False,
        "hidden_size": 1150,
        "num_layers": 3,
    }
    # text = "The colleague sitting next to me is [MASK]"

    # Initialize KGLM
    model = Kglm(**MODEL_PARAMS)

    # forward pass
    MODEL_ARGS = {
        "source": Dict[str, torch.Tensor],
        "reset": torch.Tensor,
        "metadata": List[Dict[str, Any]],
        "target": Dict[str, torch.Tensor],
        "alias_database": AliasDatabase,
        "mention_type": torch.Tensor,
        "raw_entity_ids": Dict[str, torch.Tensor],
        "entity_ids": Dict[str, torch.Tensor],
        "parent_ids": Dict[str, torch.Tensor],
        "relations": Dict[str, torch.Tensor],
        "shortlist": Dict[str, torch.Tensor],
        "shortlist_inds": torch.Tensor,
        "alias_copy_inds": torch.Tensor
    }

    outputs = model(**MODEL_ARGS)

    # See if the dataset works
    for x in di(ds.load(LOC.lw2 / 'train.jsonl'), alias_database=ds.alias_database):
        print(x)
        print('potato')
        model(**x)
        break



if __name__ == '__main__':
    main()
