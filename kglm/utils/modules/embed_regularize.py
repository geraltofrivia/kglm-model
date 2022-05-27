import torch

def embedded_dropout(embed, words, dropout=0.1, scale=None):
  if dropout:
    # mask = embed.weight.data.new().resize_((embed.weight.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed.weight) / (1 - dropout)
    mask = embed.data.new().resize_((embed.size(0), 1)).bernoulli_(1 - dropout).expand_as(embed) / (1 - dropout)

    # masked_embed_weight = mask * embed.weight
    masked_embed_weight = mask * embed

  else:
    # masked_embed_weight = embed.weight
    masked_embed_weight = embed
  if scale:
    masked_embed_weight = scale.expand_as(masked_embed_weight) * masked_embed_weight

  # try:
  #   padding_idx = embed.padding_index
  # except AttributeError:
  #   padding_idx = embed.padding_idx
  padding_idx = None

  if padding_idx is None:
      padding_idx = 0

  # X = torch.nn.functional.embedding(words, masked_embed_weight,
  #   padding_idx, embed.max_norm, embed.norm_type,
  #   embed.scale_grad_by_freq, embed.sparse
  # )
  X = torch.nn.functional.embedding(words, masked_embed_weight, padding_idx)

  return X