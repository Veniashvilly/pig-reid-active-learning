import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class PigReIDModel(nn.Module):
    def __init__(self, num_classes: int, embedding_dim: int = 512, pretrained: bool = False):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)
        backbone_output_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.embedding_layer = nn.Linear(backbone_output_dim, embedding_dim)
        self.embedding_bn = nn.BatchNorm1d(embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)


    def forward(self, images):
        features = self.backbone(images)
        embeddings = self.embedding_layer(features)
        embeddings = self.embedding_bn(embeddings)
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        logits = self.classifier(normalized_embeddings)
        return normalized_embeddings, logits


    def get_embeddings(self, images):
        embeddings, _ = self.forward(images)
        return embeddings