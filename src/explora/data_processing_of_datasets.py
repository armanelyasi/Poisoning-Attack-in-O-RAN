import pandas as pd
''
if __name__ == '__main__':

    columns_to_extract = [
        "dl_buffer [bytes]",
        "tx_brate downlink [Mbps]"]

    dataset_complete_name = ['/share/explora-dataset-pkl/dataset_new_explora_k1.pkl']

    is_save_dataset_to_pickle = False
    new_dataset_complete_name = '/home/salvo/Desktop/explora_dataset/dataset_new_explora_k12.pkl'

    dataset = [pd.read_pickle(dataset_complete_name[i]) for i in range(len(dataset_complete_name))]

    # Combine datasets into one pkl
    if len(dataset) == 1:
        dataset = dataset[0]
        print("No concat() needed!")
    else:
        dataset = pd.concat(dataset,ignore_index=True)

    # Save to pickle
    if is_save_dataset_to_pickle:
        dataset.to_pickle(new_dataset_complete_name)
        print("New pickle created!")

    # Extracting dataset stats and info
    max_stats = [dataset[columns_to_extract[i]].max() for i in range(len(columns_to_extract))]
    min_stats = [dataset[columns_to_extract[i]].min() for i in range(len(columns_to_extract))]
    mean_stats = [dataset[columns_to_extract[i]].mean() for i in range(len(columns_to_extract))]
    median_stats = [dataset[columns_to_extract[i]].median() for i in range(len(columns_to_extract))]

    for i in range(len(columns_to_extract)):
        print("KPI - {} : [{}, {}, {}, {}] (min/max/mean/median)".format(columns_to_extract[i],
                                                                         min_stats[i],
                                                                         max_stats[i],
                                                                         mean_stats[i],
                                                                         median_stats[i]))
