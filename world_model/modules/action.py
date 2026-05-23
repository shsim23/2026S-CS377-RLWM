import torch.nn as nn


class ActionEmbedder(nn.Module):
    def __init__(self, num_actions: int = 5, emb_dim: int = 32):
        super().__init__()
        self.embedding = nn.Embedding(num_actions, emb_dim)

    def forward(self, a):
        return self.embedding(a)
