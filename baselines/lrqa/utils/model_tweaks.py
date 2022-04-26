import transformers


def adjust_tokenizer(tokenizer):
    if isinstance(tokenizer, (transformers.GPT2Tokenizer, transformers.GPT2TokenizerFast)) and \
            "gpt" in tokenizer.name_or_path:
        tokenizer.pad_token = tokenizer.eos_token
