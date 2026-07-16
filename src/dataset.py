from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset


class PigReIDDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe.reset_index(drop=True).copy()
        self.transform = transform
        
    def __len__(self):
        return len(self.dataframe)
    

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        image_path = Path(row["image_path"])
        label = int(row["label_idx"])
        identity_id = str(row["identity_id"])
        image = Image.open(image_path)
        image = image.convert("RGB")
        
        if self.transform is not None:
            image = self.transform(image)
        return {
            "image": image,
            "label": label,
            "identity_id": identity_id,
            "image_path": str(image_path)
        }