"""Messing around with GPT"""
# https://huggingface.co/docs/transformers/v4.22.2/en/model_doc/gpt2

# imports
from transformers import GPT2LMHeadModel, GPT2Model, GPT2Tokenizer
from transformers import pipeline, set_seed


def text_generation_1(
        sentence: str
):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2', pad_token_id=tokenizer.eos_token_id)
    input_ids = tokenizer.encode(sentence, return_tensors='pt')
    output = model.generate(input_ids,
                        max_length=100,
                        num_beams=5,
                        no_repeat_ngram_size=2,
                        early_stopping=True)
    print(tokenizer.decode(output[0], skip_special_tokens=True))

def text_generation_2(
        sentence: str
):
    generator = pipeline('text-generation', model='gpt2')
    set_seed(42)
    print(generator(sentence, max_length=3, num_return_sequences=5))

def get_features_from_text(
        sentence: str
):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    encoded_input = tokenizer(sentence, return_tensors='pt')
    output = model(**encoded_input)


text_generation_1("What is deep learning?")
text_generation_2("The colleague sitting next to me is [MASK].")