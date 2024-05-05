# Function to merge similar dataset labels
# Team members:
# - Kevin Tao
# - Haneul Lee

# Description:
# Dataset labels are mapped to a more general category to reduce the number of classes in the dataset.
# Also, only the first 50k samples are selected, due to issues with the dataset content after this point.

# System specifications during testing:
# Windows 11 Home 23H2 OS Build 22631.3447
# AMD Ryzen 7 5700 @ 3.7 GHz, 32 GB RAM DDR4 @ 3200 MHz CL16
# NVIDIA GeForce RTX 4060 Ti 8 GB GDDR6 DX12

def merge_dataset(dataset):
    labelMap = {
        'high energy physics theory': 'theoretical physics',
        'quantum physics': 'theoretical physics',
        'high energy physics phenomenology': 'theoretical physics',
        'condensed matter': 'general physics',
        'physics': 'general physics',
        'astrophysics': 'general physics',
        'mathematics': 'general mathematics',
        'statistics': 'general mathematics',
    }
    dataset = dataset.select([i for i in range(50000)]) # Select only the first 50k samples
    dataset = dataset.map(lambda sample: {**sample, 'label': labelMap.get(sample["label"], sample["label"])}) # Merge labels
    
    return dataset
