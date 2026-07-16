from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from reid_model import PigReIDModel


class PigReIDEmbedder:
    def __init__(
        self,
        checkpoint_path,
        device=None,
        input_height=256,
        input_width=128,
        embedding_dim=512,
        num_classes=None
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)

        # Поддерживаем два варианта:
        # 1) полный checkpoint с ключом "model_state_dict";
        # 2) просто state_dict.
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

            self.num_classes = checkpoint.get("num_classes", num_classes)
            self.embedding_dim = checkpoint.get("embedding_dim", embedding_dim)

            self.input_height = checkpoint.get("input_height", input_height)
            self.input_width = checkpoint.get("input_width", input_width)

            self.mean = checkpoint.get("mean", [0.485, 0.456, 0.406])
            self.std = checkpoint.get("std", [0.229, 0.224, 0.225])

        else:
            state_dict = checkpoint

            # Если дали только state_dict, пробуем автоматически понять размеры.
            self.embedding_dim = state_dict["embedding_layer.weight"].shape[0]
            self.num_classes = state_dict["classifier.weight"].shape[0]

            self.input_height = input_height
            self.input_width = input_width

            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

        if self.num_classes is None:
            raise ValueError("num_classes is None. Use full checkpoint or pass num_classes manually.")

        self.model = PigReIDModel(
            num_classes=self.num_classes,
            embedding_dim=self.embedding_dim,
            pretrained=False
        )

        self.model.load_state_dict(state_dict)

        self.model = self.model.to(self.device)

        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.mean,
                std=self.std
            )
        ])

    def preprocess_crop(self, crop_bgr):
        if not isinstance(crop_bgr, np.ndarray):
            raise TypeError("crop_bgr must be numpy.ndarray")

        if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
            raise ValueError("crop_bgr must have shape [H, W, 3]")

        if crop_bgr.shape[0] == 0 or crop_bgr.shape[1] == 0:
            raise ValueError("empty crop was passed to ReID model")
        crop_rgb = crop_bgr[:, :, ::-1].copy()

        image = Image.fromarray(crop_rgb)

        image = image.resize(
            (self.input_width, self.input_height),
            Image.Resampling.LANCZOS
        )

        tensor = self.transform(image)

        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(self.device)

        return tensor

    def get_embedding(self, crop_bgr):
        tensor = self.preprocess_crop(crop_bgr)

        with torch.no_grad():
            embedding = self.model.get_embeddings(tensor)
            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        return embedding.squeeze(0).cpu().numpy()

    def get_embedding_tensor(self, crop_bgr):
        tensor = self.preprocess_crop(crop_bgr)

        with torch.no_grad():
            embedding = self.model.get_embeddings(tensor)

            embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)

        return embedding.squeeze(0)