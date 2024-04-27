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
