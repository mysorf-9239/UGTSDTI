import torch
import torch.nn as nn
import torch.nn.functional as F

from ugtsdti.core.registry import MODELS


@MODELS.register("cnn1d_student")
class CNN1DStudent(nn.Module):
    """
    A simple 1D CNN baseline student model for DTI sequence inputs.
    """

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int, num_classes: int = 1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size=3, padding=1)

        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        # x shape: (Batch, SequenceLength) -> Simplified mock input handling
        # Real logic would extract sequences from a dict
        if isinstance(x, dict):
            # mock extraction
            seq = next(iter(x.values()))
        else:
            seq = x

        emb = self.embedding(seq).transpose(1, 2)  # (Batch, Embed, SeqLen)

        h = F.relu(self.conv1(emb))
        h = F.relu(self.conv2(h))

        # Global max pooling
        h, _ = torch.max(h, dim=2)

        logits = self.fc(h).squeeze(-1)
        return torch.sigmoid(logits)
