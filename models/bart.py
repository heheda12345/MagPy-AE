from transformers import AutoTokenizer, AutoConfig, AutoModel
import torch

# low speedup

model_name = "microsoft/deberta-base"
device = "cuda:0"

def get_model():
    config = AutoConfig.from_pretrained(model_name)
    config.return_dict = True
    model = AutoModel.from_config(config).to(device)
    print(model)
    return model


def get_input(batch_size):
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # inputs = tokenizer("Hello world! Hello world! Hello world! Hello world! Hello world!", return_tensors="pt").to(device)
    # print(inputs)
    # return (inputs['input_ids'], inputs['attention_mask'], inputs['token_type_ids']), {}
    vocab_size = 50265
    seq_len = 256
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), dtype=torch.int64).to(device)
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.int64).to(device)
    token_type_ids = torch.zeros((batch_size, seq_len), dtype=torch.int64).to(device)
    return (input_ids, attention_mask, token_type_ids), {}


if __name__ == "__main__":
    model = get_model()
    input_args, input_kwargs = get_input(batch_size=1)
    print([x.shape for x in input_args])
    outputs = model(*input_args, **input_kwargs)
    print(outputs)

