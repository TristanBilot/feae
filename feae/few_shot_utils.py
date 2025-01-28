from collections import defaultdict

import torch

def get_few_shot_indices(N, percentage_of_benign, train_labels, train_attack_families):
    count_attacks_dict = defaultdict(int) 
    global_count_attacks = 0
    indices, mal_indices = [], []

    for i, (label, attack) in enumerate(zip(train_labels, train_attack_families)):
        if label == 1:
            if count_attacks_dict[attack] < N:
                count_attacks_dict[attack] += 1
                global_count_attacks += 1
                indices.append(i)
                mal_indices.append(i)
        else:
            if i < len(train_labels) * percentage_of_benign:
                indices.append(i)

    return (
        torch.tensor(indices, dtype=int), 
        torch.tensor(mal_indices, dtype=int),
    )
