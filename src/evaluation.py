import torch
from torch.utils.data import DataLoader
from torchmetrics.retrieval import RetrievalMAP, RetrievalPrecision

def extract_embeddings(model, dataloader: DataLoader, device):
    model.eval()
    all_embeddings = []
    all_labels = []
    all_identity_ids = []
    all_image_paths = []
    
    with torch.no_grad():
        
        for batch in dataloader:
            images = batch["image"]
            labels = batch["label"]
            images = images.to(device)
        
            embeddings = model.get_embeddings(images)
            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())
            all_identity_ids.extend(batch["identity_id"])
            all_image_paths.extend(batch["image_path"])
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    return all_embeddings, all_labels, all_identity_ids, all_image_paths


def compute_rank1_map(query_embeddings, query_labels, gallery_embeddings, gallery_labels):
    query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=1)
    gallery_embeddings = torch.nn.functional.normalize(gallery_embeddings, p=2, dim=1)
    
    similarity_matrix = query_embeddings @ gallery_embeddings.T
    num_queries = similarity_matrix.shape[0]
    num_gallery = similarity_matrix.shape[1]
    
    preds = similarity_matrix.reshape(-1)
    
    target_matrix = query_labels.unsqueeze(1) == gallery_labels.unsqueeze(0)
    target = target_matrix.reshape(-1)
    
    indexes = torch.arange(num_queries).repeat_interleave(num_gallery)
    
    rank1_metric = RetrievalPrecision(top_k=1)
    map_metric = RetrievalMAP()
    rank1 = rank1_metric(preds, target, indexes=indexes)
    
    mean_average_precision = map_metric(preds, target, indexes=indexes)

    return {
        "rank1": rank1.item(),
        "mAP": mean_average_precision.item()
    }