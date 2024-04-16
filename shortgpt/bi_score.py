import torch
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaTokenizer, MistralForCausalLM
import bitsandbytes, flash_attn

def calculate_block_influence(model, inputs):
    model.eval()
    influences = []

    with torch.no_grad():
        outputs = model(inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states 

        for i in range(len(hidden_states) - 1):
            if i + 1 < len(hidden_states):
                state_i, state_j = hidden_states[i], hidden_states[i + 1]
                influence = 1 - torch.mean(torch.sum(state_j * state_i, dim=-1) /
                                           (torch.norm(state_i, dim=-1) * torch.norm(state_j, dim=-1)))
                influences.append(influence.item())

    return influences

model_name = "NousResearch/Hermes-2-Pro-Mistral-7B"
tokenizer = LlamaTokenizer.from_pretrained(model_name)
model = MistralForCausalLM.from_pretrained(model_name, load_in_8bit=False,torch_dtype=torch.float16,
    device_map="auto",
    load_in_4bit=True,
    use_flash_attention_2=False)

input_tokens = tokenizer.encode("<|im_start|>system\nYou are Hermes 2.<|im_end|>\n<|im_start|>user\nHello, who are you?<|im_end|>\n<|im_start|>assistant", return_tensors="pt")
data_loader = [input_tokens]

def visualize_bi_scores(bi_scores):
    num_layers = len(bi_scores)

    plt.figure(figsize=(10, 6))
    plt.bar(range(num_layers), bi_scores)
    plt.xlabel('Layer Index')
    plt.ylabel('Block Influence Score')
    plt.title('Block Influence Scores')
    plt.show()

bi_scores = calculate_block_influence(model, input_tokens[0].unsqueeze(0))
visualize_bi_scores(bi_scores)
