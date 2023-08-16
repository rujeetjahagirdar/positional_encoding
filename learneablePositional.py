import torch
import torch.nn as nn

# Define the Learnable Positional Encoding model (from our previous code)
class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(LearnablePositionalEncoding, self).__init__()
        self.positional_embedding = nn.Embedding(max_len, d_model)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        pos_enc = self.positional_embedding(positions)
        return x + pos_enc

# Sample Sentence
# sentence = "The quick brown fox jumps over the lazy dog."
# sentence = "This is a sample statement for positional embedding."
sentence  = input("Enter the sentence:\n")
tokens = sentence.split()
# print("Tokens:\n",tokens)


vocab = {word: i for i, word in enumerate(set(tokens))}
# print("Vocab:\n",vocab)
token_ids = [vocab[word] for word in tokens]
# print("Token_ids\n",token_ids)

# Convert token_ids to PyTorch tensor
input_tensor = torch.tensor(token_ids).unsqueeze(0)  # Shape [1, sequence_length]
# print("Input_tensor\n",input_tensor)

# Create an embedding layer
embedding_dim = 128
vocab_size = len(vocab)
embedding = nn.Embedding(vocab_size, embedding_dim)
embedded_tokens = embedding(input_tensor)
# print("embedded_tokens\n",embedded_tokens)

# Positional Encoding
pos_encoder = LearnablePositionalEncoding(embedding_dim)
positionally_encoded_tokens = pos_encoder(embedded_tokens)

print(positionally_encoded_tokens)
print(positionally_encoded_tokens.shape)
