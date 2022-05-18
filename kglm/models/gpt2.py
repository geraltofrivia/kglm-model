# from transformers import GPT2Tokenizer, GPT2Model
#
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2Model.from_pretrained('gpt2')
# text = "The colleague sitting next to me is [MASK]."
# encoded_input = tokenizer(text, return_tensors='pt')
# output = model(**encoded_input)
#
# debug = 0

''' Text generation '''
# from transformers import pipeline, set_seed
# generator = pipeline('text-generation', model='gpt2')
# set_seed(42)
# print(generator(text, max_length=30, num_return_sequences=5))

# https://huggingface.co/docs/transformers/model_doc/gpt2
from transformers import GPT2Model, GPT2Config, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
configuration = GPT2Config()
model = GPT2Model(configuration)
# configuration = model.config

text = "The colleague sitting next to me is [MASK]."
inputs = tokenizer(text, return_tensors="pt")
outputs = model(**inputs)
last_hidden_states = outputs.last_hidden_state