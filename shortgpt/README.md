
[ShortGPT: Layers in Large Language Models are More Redundant Than You Expect](https://arxiv.org/abs/2403.03853)

Block Influence(BI) scores are a measure of how much each layer influences the next layer's activations, providing insights into the flow of information through the model. THe function iterates over the hidden states, comparing each state with the next state using a formula that calculates the cosine similarity between the states. Higher values indicate a greater influence of the previous layer on the current layer. The function returns a list of scores.
