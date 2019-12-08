import math
def batches(batch_size,features,labels):
    """
    Create batches of features and labels
    :param batch_size: The batch size
    :param features: List of features
    :param babels: List of labels
    :return: Batches of (Features,Labels)
    """
    assert len(features) == len(labels)

    # implement batching
    output_batches = []

    for start_i in range(0,len(features),batch_size):
        end_i = start_i + batch_size
        batch = [[features[start_i:end_i]],[labels[start_i:end_i]]]
        output_batches.append(batch)

    return output_batches
