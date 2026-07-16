import torch


def core_set_selection(labeled_embeddings, unlabeled_embeddings, budget):
    labeled_embeddings = torch.nn.functional.normalize(labeled_embeddings, p=2, dim=1)
    unlabeled_embeddings = torch.nn.functional.normalize(unlabeled_embeddings, p=2, dim=1)
    
    similarity = unlabeled_embeddings @ labeled_embeddings.T
    distance = 1.0 - similarity
    min_distances, _ = distance.min(dim=1)
    
    selected_indices = []
    
    available_mask = torch.ones(unlabeled_embeddings.shape[0], dtype=torch.bool)
    
    for i in range(budget):
        candidate_distances = min_distances.clone()

        candidate_distances[~available_mask] = -1.0
    
        selected_index = torch.argmax(candidate_distances).item()
        
        selected_indices.append(selected_index)
        
        available_mask[selected_index] = False
    
        selected_embedding = unlabeled_embeddings[selected_index:selected_index + 1]
        
        new_similarity = unlabeled_embeddings @ selected_embedding.T
        
        new_distance = 1.0 - new_similarity.squeeze(1)
        
        min_distances = torch.minimum(min_distances, new_distance)
    
    return selected_indices