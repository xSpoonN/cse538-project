
def map_labels(sample):
    label = sample["label"]

    # theoretical physics
    if label == 'high energy physics theory' or label == 'quantum physics':
        sample["label"] = 'theoretical physics'
    # general physics
    elif label == 'condensed matter' or label == 'physics' or label == 'astrophysics':
        sample["label"] = 'general physics'
    # general mathmetics
    elif label == 'mathematics' or label == 'statistics':
        sample["label"] = 'general mathematics'
    return sample

def merge_dataset(dataset):
    # find labels which is useful
    label_dic = {}
    for data in dataset:
        label = data['label']
        if label in label_dic.keys():
            label_dic[label] += 1
        else:
            label_dic[label] = 1
    keys = list(label_dic.keys())
    for label in keys:
        if label_dic[label] == 1:
            del label_dic[label]

    # remove useless datas
    selected_idx = []
    size = len(dataset)
    for i in range(size):
        label = dataset[i]['label']
        if label in label_dic.keys():
            selected_idx.append(i)
            continue
    dataset = dataset.select(selected_idx)


    # merging labels
    dataset = dataset.map(map_labels)
    
    return dataset
