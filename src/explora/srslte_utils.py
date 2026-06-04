import math
import numpy as np
import os

import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import glob

import tensorflow as tf
import pickle
# from tensorflow.keras.layers.experimental import preprocessing
from autoencoder_model import AutoencoderModel
import config_file


def entire_dataset_from_single_file(filename,
                                    col_names,
                                    selected_col_names,
                                    remove_zero_req_prb_entries=False,
                                    remove_zero_throughput_entries=False,
                                    remove_zero_cqi_entries=False,
                                    remove_zero_mcs_entries=False,
                                    scale_dl_buffer=False,
                                    replace_zero_with_one=False,
                                    add_prb_ratio=True):

    dataset = pd.read_csv(filename, names=col_names, usecols=selected_col_names, header=0)

    if remove_zero_req_prb_entries:
        dataset = dataset.loc[dataset['sum_requested_prbs'] > 0].reset_index(drop=True)

    if remove_zero_throughput_entries:
        dataset = dataset.loc[dataset['tx_brate downlink [Mbps]'] > 0].reset_index(drop=True)

    if remove_zero_cqi_entries:
        dataset = dataset.loc[dataset['dl_cqi'] >= 1].reset_index(drop=True)

    if remove_zero_mcs_entries:
        dataset = dataset.loc[dataset['dl_mcs'] >= 1].reset_index(drop=True)
        # dataset = dataset.loc[dataset['ul_mcs'] >= 1].reset_index(drop=True)

    if scale_dl_buffer and any(["dl_buffer [bytes]" in m for m in
                                selected_col_names]):  # we scale the dl_buffer which is way larger than the others
        dataset['dl_buffer [bytes]'] = dataset['dl_buffer [bytes]'] / 100000

    if add_prb_ratio:
        dict_add = pd.DataFrame.from_dict({"ratio_req_granted": np.clip(np.nan_to_num(
            dataset["sum_granted_prbs"] / dataset["sum_requested_prbs"]), a_min=0, a_max=1)
        })
        if replace_zero_with_one:
            dict_add['ratio_req_granted'].loc[dataset['sum_requested_prbs'] <= 0] = 1.0
        return dataset.join(dict_add)
    else:
        return dataset


# returns all csv files inside a single DataFrame
def entire_dataset_from_folder(main_folder,
                               wildcard,
                               col_names,
                               selected_col_names,
                               scale_dl_buffer=False,
                               remove_zero_req_prb_entries=False,
                               remove_zero_throughput_entries=False,
                               remove_zero_cqi_entries=False,
                               remove_zero_mcs_entries=False,
                               replace_zero_with_one=False,
                               add_prb_ratio=True):
    dataset = []
    folders_list = glob.glob(main_folder + wildcard)
    n_folders = len(folders_list)
    i = 1
    for filename in folders_list:
        print("Processing folder {} ({}/{}) ".format(filename,i,n_folders))
        db_tmp = entire_dataset_from_single_file(filename, col_names=col_names,
                                                 selected_col_names=selected_col_names,
                                                 scale_dl_buffer=scale_dl_buffer,
                                                 remove_zero_req_prb_entries=remove_zero_req_prb_entries,
                                                 remove_zero_throughput_entries=remove_zero_throughput_entries,
                                                 replace_zero_with_one=replace_zero_with_one,
                                                 remove_zero_cqi_entries=remove_zero_cqi_entries,
                                                 remove_zero_mcs_entries=remove_zero_mcs_entries,
                                                 add_prb_ratio=add_prb_ratio)
        dataset.append(db_tmp)
        i += 1
        print("Added {} rows".format(len(db_tmp)))

    return pd.concat(dataset, axis=0, ignore_index=True)


# this takes the most updated entries from the DataFrame by pulling from the tail
def extract_n_entries_from_bottom_per_slice(dataset=None,
                                            slice_id=None,
                                            chunk_size=10,
                                            metrics_export=None):
    if slice_id is not None:
        d_temp = dataset.loc[dataset['slice_id'] == int(slice_id)]
    else:
        d_temp = dataset

    d_temp = d_temp.tail(n=chunk_size).reset_index(drop=True)
    if metrics_export is not None:
        d_temp = d_temp[metrics_export]

    return d_temp

def extract_observation_from_dataset(dataset=None,
                                     slice_ids=None,
                                     slice_prb=None,
                                     scheduling_policy=None,
                                     metrics_export=None,
                                     sample=True,
                                     n_sample=None):
    # TODO: here metrics_export must be metrics_export[i]
    return [extract_dataset_for_training(dataset=dataset,
                                         slice_id=None if slice_ids is None else slice_ids[i],
                                         slice_prb=None if slice_prb is None else slice_prb[i],
                                         scheduling_policy=None if scheduling_policy is None else scheduling_policy[i],
                                         metrics_export=metrics_export[i],
                                         sample=sample,
                                         n_sample=n_sample[i]) for i in range(len(slice_ids))]

# Input: DataFrame (ideally taken from entire_dataset_from_folder())
# Output: all entries in the dataset with specific slice_id, slice_prb, and sched_policy. You get metrics_export columns only. Output is DataFrame.
def extract_dataset_for_training(dataset=None,
                                 slice_id=None,
                                 slice_prb=None,
                                 scheduling_policy=None,
                                 metrics_export=None,
                                 sample=False,
                                 n_sample=0):
    if slice_id is not None:
        d_temp = dataset.loc[dataset['slice_id'] == slice_id]
    else:
        d_temp = dataset

    if slice_prb is not None:
        d_temp = d_temp.loc[d_temp['slice_prb'] == int(slice_prb)]

    if scheduling_policy is not None:
        d_temp = d_temp.loc[d_temp['scheduling_policy'] == int(scheduling_policy)]

    if metrics_export is not None:
        d_temp = d_temp[metrics_export]

    if sample and n_sample > 0:
        return d_temp.sample(n_sample)
    else:
        return d_temp

# Input: DataFrame (ideally taken from entire_dataset_from_folder())
# Output: all entries in the dataset with specific slice_id, slice_prb, and sched_policy. You get metrics_export columns only. Output is DataFrame.
def extract_dataset_for_training_v2(dataset=None,
                                    slice_ids=None,
                                    slice_prbs=None,
                                    scheduling_policy=None,
                                    metrics_export=None,
                                    sample=False,
                                    n_sample=0):
    if slice_ids is not None:
        # TODO: this sometimes load duplicate data, we remove them after this step, but it would be nice to avoid loading duplicates here
        d_temp = [dataset.loc[dataset['slice_id'] == slice_id] for slice_id in slice_ids]

        # extract only prb_allocations that make sense
        if slice_prbs is not None:
            dataset_per_slice = list()
            for s in range(len(slice_ids)):
                unique_prbs_setting_slice = np.unique(np.array(slice_prbs[0:])[:,s])
                subentries = list()
                for i in range(len(unique_prbs_setting_slice)):
                    subentries.append(
                        d_temp[s].loc[d_temp[s]['slice_prb'] == unique_prbs_setting_slice[i]]
                    )
                dataset_per_slice.append(
                    pd.concat(subentries, axis=0, ignore_index=False)
                )
            d_temp = pd.concat(dataset_per_slice, axis=0, ignore_index=False)
        else:
            d_temp = pd.concat(d_temp, axis=0, ignore_index=False)
    else:
        d_temp = dataset

    # remove duplicated index entries in case we extracted some more than once
    d_temp = d_temp[~d_temp.index.duplicated()].reset_index(drop=True)

    if scheduling_policy is not None:
        d_temp = d_temp.loc[d_temp['scheduling_policy'] == int(scheduling_policy)]

    if metrics_export is not None:
        d_temp = d_temp[metrics_export]

    if sample is True and n_sample > 0:
        return d_temp.sample(n_sample)
    else:
        return d_temp

def train_autoencoder(bottleneck_dim=10,
                      n_entries_per_sample=4,
                      n_metrics=4,
                      learning_rate=0.001,
                      n_epochs=100,
                      train_dataset=None,
                      val_dataset=None,
                      checkpoint_folder=None,
                      patience=10,
                      save_path=None,
                      is_regularized=True):
    autoenc = AutoencoderModel(bottleneck_dim=bottleneck_dim,
                               n_entries=n_entries_per_sample,
                               n_metrics=n_metrics,
                               is_regularized=is_regularized)

    loss = tf.keras.losses.mean_squared_error
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    autoenc.compile(optimizer, loss)

    ##############DUMMY TRAINING
    # train_dataset = np.random.rand(10000,n_entries_per_sample,8)
    # val_dataset = np.random.rand(500,n_entries_per_sample,8)

    autoenc.fit(x=train_dataset,
                y=train_dataset,
                epochs=n_epochs,
                shuffle=True,
                validation_data=(val_dataset, val_dataset),
                callbacks=[
                    tf.keras.callbacks.ModelCheckpoint(checkpoint_folder, monitor='val_loss', verbose=1,
                                                       save_best_only=True,
                                                       mode='min'),
                    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience, verbose=1,
                                                     mode='min'),
                    autoenc_tensorboard_callback
                ]
                )

    autoenc.encoder.save(os.path.join(save_path, config_file.date_now + '_encoder.h5'), save_format='h5')
    autoenc.decoder.save(os.path.join(save_path, config_file.date_now + '_decoder.h5'), save_format='h5')

    return autoenc

def scale_input_to_autoencoder(X_min=None,
                               X_max=None,
                               scale=None,
                               X=None):
    return [scale * (X[i] - np.array(X_min)) / (np.array(X_max) - np.array(X_min)) for i in range(len(X))]

# The kpi_names are related to the values in X_min, X_max. X is a DataFrame with the KPI name on each column.
def scale_input_to_autoencoder_with_names(X_min=None,
                                          X_max=None,
                                          scale=None,
                                          kpi_names=None,
                                          X=None):
    X_temp = X
    for i in range(len(X)):
        for k in range(len(kpi_names)):
            X_temp[i][kpi_names[k]] = scale * (X[i][kpi_names[k]] - np.array(X_min[k])) / (np.array(X_max[k]) - np.array(X_min[k]))
    return  X_temp


def convert_prb_to_rgb(prb_allocation: list, du_prb: int=50) -> list:

    # size of RBG in PRBs (Table 7.6.1.1.1 - Type 0)
    rbg_size = 1
    if du_prb >= 11 and du_prb <= 26:
        rbg_size = 2
    elif du_prb >= 27 and du_prb <= 63:
        rbg_size = 3
    elif du_prb >= 64 and du_prb <= 110:
        rbg_size = 4
    else:
        logging.error('PRBs not supported ' + str(du_prb))
        return prb_allocation

    rbg_allocation = [math.ceil(x / rbg_size) for x in prb_allocation]
    return  rbg_allocation


if __name__ == '__main__':

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # model will be trained on GPU 0

    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(gpu_devices))
    if len(gpu_devices):
        tf.config.experimental.set_memory_growth(gpu_devices[0], True)

    is_plot = True
    dataset_shuffled = False

    is_train_autoencoder = False
    is_regularized = False
    is_load_dataset_from_csv = False # load dataset from csv
    is_load_dataset_from_pkl = False
    save_complete_dataset_to_pkl = True # save loaded dataset to pkl
    save_autoenc_dataset_to_npy = True  # save autoencoder only dataset to npy

    dir_main = "/home/globecom_paper_training_files"
    logdir = dir_main + "/autoencoder_files/logs/scalars/" + config_file.date_now + "/"
    autoenc_checkpoint_folder = dir_main + '/autoencoder_files/checkpoints/' + config_file.date_now
    autoenc_checkpoint_name = '/checkpoint.{epoch:02d}-{val_loss:.2f}.h5'
    autoenc_savepath = dir_main + '/autoencoder_files/saved_models/' + config_file.date_now + '/'
    autoenc_tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    dataset_autoenc_only_name = dir_main + '/dataset_complete_autoencoder_only.npy'
    dataset_complete_name = dir_main + '/dataset_complete.pkl'

    ###########################
    ###########################
    ######Load Whole data######
    ###########################
    ###########################
    csv_dataset_main_folder = '/home/salvo/Documents/cellular_dataset_colosseum/datasets/oran_training_50prb-main/training_data'
    wildcard_match = '/*/*/*/*/*/metrics/csv/*.csv'

    # not used now
    # slice_id = 0

    n_epochs = 30
    patience = 100000
    n_entries_per_sample = 10
    total_rows_to_extract = 10000000000000  # in case it exceeds the number of available rows, you get the min(available_rows, total_rows_to_extract)

    bottleneck_dim = config_file.bottleneck_dim

    if is_load_dataset_from_csv:
        dataset_original = entire_dataset_from_folder(main_folder=csv_dataset_main_folder,
                                                      wildcard=wildcard_match,
                                                      col_names=config_file.metrics_list,
                                                      selected_col_names=config_file.metric_list_to_extract,
                                                      remove_zero_throughput_entries=config_file.remove_zero_throughput_entries,
                                                      remove_zero_req_prb_entries=config_file.remove_zero_req_prb_entries,
                                                      add_prb_ratio=config_file.add_prb_ratio)

        if save_complete_dataset_to_pkl:
            dataset_original.to_pickle(dataset_complete_name)

        dataset_original = extract_n_entries_from_bottom_per_slice(dataset=dataset_original,
                                                                   slice_id=None,
                                                                   chunk_size=total_rows_to_extract,
                                                                   metrics_export=config_file.metric_list_autoencoder)

        # preparing dataset as numpy arrays
        dataset = dataset_original.to_numpy()

        if save_autoenc_dataset_to_npy:
            with open(dataset_autoenc_only_name, 'wb') as f:
                np.save(f,dataset)
    elif is_load_dataset_from_pkl:
        dataset_original = pickle.load( open( dataset_complete_name, "rb" ) )
        dataset_original = extract_n_entries_from_bottom_per_slice(dataset=dataset_original,
                                                                   slice_id=None,
                                                                   chunk_size=total_rows_to_extract,
                                                                   metrics_export=config_file.metric_list_autoencoder)

        # preparing dataset as numpy arrays
        dataset = dataset_original.to_numpy()

        if save_autoenc_dataset_to_npy:
            with open(dataset_autoenc_only_name, 'wb') as f:
                np.save(f,dataset)

    else:
        with open(dataset_autoenc_only_name, 'rb') as f:
            dataset = np.load(f)
        print("dataset loaded")
        dataset_original = pd.read_pickle(dataset_complete_name)

    dataset_size = int(dataset.shape[0] / n_entries_per_sample) * n_entries_per_sample  # total number of batched samples we can get from the dataset
    dataset = dataset[0:dataset_size, :]  # getting rid of unpaired entries
    n_tot_samples = int(dataset_size / n_entries_per_sample)
    if dataset_shuffled:
        np.random.shuffle(dataset)

    # do some scaling
    # this is what it does
    # X_min = np.array([  0.,  0. , 0.     ]) for oran_training_50prb-main
    # X_max = np.array([  1.84665,  13.8804 , 571.     ]) oran_training_50prb-main
    # X_scaled = (X - X_min) / (X_max - X_min)
    # X_original = X_std * (X_max - X_min) + X_min
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaler.fit(dataset)
    dataset = scaler.transform(dataset)
    dataset = 10 * dataset

    # reshape according to batch_size = n_entries_per_sample
    dataset = dataset.reshape((n_tot_samples, n_entries_per_sample, len(config_file.metric_list_autoencoder)))

    train_size = int(0.7 * n_tot_samples)
    val_size = int(0.15 * n_tot_samples)
    test_size = int(0.15 * n_tot_samples)

    train_dataset = dataset[0:train_size, :, :]
    test_dataset = dataset[train_size:(train_size + test_size), :, :]
    val_dataset = dataset[train_size + test_size:, :, :]

    print("test val created")

    if config_file.simulate_zero_entries:
        train_dataset = np.concatenate((train_dataset,np.zeros(train_dataset.shape)))
        test_dataset = np.concatenate((test_dataset, np.zeros(test_dataset.shape)))
        val_dataset = np.concatenate((val_dataset, np.zeros(val_dataset.shape)))

    # THIS IS NOT REALLY USED, BUT MIGHT BE USEFUL IN THE FUTURE
    # prepare dataset as tensors, also use batch here since later we use Flatten in the autoencoder
    # dataset_all = tf.data.Dataset.from_tensor_slices(dataset)
    # dataset_all = dataset_all.shuffle(n_samples).batch(n_entries_per_sample,drop_remainder=True)
    # train_dataset = dataset_all.take(train_size)
    # test_dataset = dataset_all.skip(train_size)
    # val_dataset = test_dataset.skip(test_size)
    # test_dataset = test_dataset.take(test_size# )
    # x_train = train_dataset.map(lambda x: (x, x))
    # x_val = val_dataset.map(lambda x: (x,x))
    # autoenc.fit(x_train,
    #             epochs=n_epochs,
    #             shuffle=True,
    #             validation_data=x_val
    #             )

    print(train_dataset.shape)

    if is_train_autoencoder:

        import shutil
        os.mkdir(autoenc_savepath)
        os.mkdir(autoenc_checkpoint_folder)
        shutil.copy('./config_file.py', autoenc_savepath)
        shutil.copy('./srslte_utils.py', autoenc_savepath)
        shutil.copy('./autoencoder_model.py', autoenc_savepath)

        autoenc = train_autoencoder(bottleneck_dim=bottleneck_dim,
                                    n_entries_per_sample=n_entries_per_sample,
                                    n_metrics=len(config_file.metric_list_autoencoder),
                                    learning_rate=0.001,
                                    n_epochs=n_epochs,
                                    train_dataset=train_dataset,
                                    val_dataset=val_dataset,
                                    checkpoint_folder=autoenc_checkpoint_folder + autoenc_checkpoint_name,
                                    patience=patience,
                                    save_path=autoenc_savepath,
                                    is_regularized=is_regularized)

    else: # just testing with already trained model
        encoder_filename = './ml_models/encoder.h5'
        decoder_filename = './ml_models/decoder.h5'
        encoder = tf.keras.models.load_model(encoder_filename)
        decoder = tf.keras.models.load_model(decoder_filename)

        n_test_entries = 50000

        np.random.shuffle(test_dataset)

        x = test_dataset[0:n_test_entries,:,:].astype('float32')

        print(test_dataset)
        print(test_dataset.shape)

        all_zero_entry = 0
        for entry in test_dataset:
            sum = 0
            for row in entry:
                for col in row:
                    sum = sum + col
            if sum == 0:
                all_zero_entry += 1

        print("Found %d all zeros 10x3 entries" % all_zero_entry)

        enc_x = encoder.predict(x).astype('float32')
        dec_x = decoder.predict(enc_x).astype('float32')

        from sklearn.metrics import mean_squared_error

        x_flatten = np.reshape(x,(n_test_entries,30))
        dec_x_flatten = np.reshape(dec_x, (n_test_entries, 30))

        mse = mean_squared_error(x_flatten,dec_x_flatten)
        print(mse)

    ###########################
    ###########################
    ######Load batch data######
    ###########################
    ###########################
    # main_folder = '/home/salvo/Documents/cellular_dataset_colosseum/ml/ml_scheduling_slicing'
    # wildcard_match = '/*/*/metrics/csv/*.csv'
    #
    # dataset = bottom_batches_from_folders(main_folder=main_folder,wildcard=wildcard_match,batch_size=16,n_batches=1)

    ###########################
    ###########################
    ####Load specific file#####
    ###########################
    ###########################
    # filename = '/home/salvo/Documents/cellular_dataset_colosseum/ml/ml_scheduling_slicing/exp5/srs_config_bs1/metrics/csv/1010123456003_metrics.csv'
    # dataset = dataset_from_single_file(filename=filename)

    ###########################
    ###########################
    ####Plotting for testing###
    ###########################
    ###########################

    if is_plot:
        # plot hist if needed
        dataset[config_file.metric_list_autoencoder].hist(bins=50, figsize=(20, 15))

        # plot corr
        scatter_matrix(dataset[config_file.metric_list_autoencoder], figsize=(20, 15))
        plt.savefig('autoenc.png')
        plt.show()

    # def show_batches(dataset, n_batches=1):
    #     for batch in dataset.take(n_batches):
    #         for key, value in batch.items():
    #             print("{:20s}: {}".format(key, value.numpy()))
    #
    # def get_csv_rows_number(filename, chunk_size=1000):
    #     n_lines = 0
    #     for chunk in pd.read_csv(filename, chunksize=chunk_size,
    #                              usecols=[7]):  # usecols=1 as that is an integer (i.e., slice_id) and faster to read
    #         n_lines += chunk.shape[0]
    #     return n_lines
    #
    #
    # def get_bottom_rows_from_file_large(filename,
    #                                     chunksize=1000):  # this is a version that works best for very large files
    #     tot_n_rows = get_csv_rows_number(filename, chunk_size=chunksize)
    #     db_tmp = pd.read_csv(filename, nrows=chunksize + 1,
    #                          skiprows=range(1, tot_n_rows - chunksize + 1))  # this should read the last chunk_size rows
    #     dict_add = pd.DataFrame.from_dict({"ratio_req_granted": np.nan_to_num(
    #         np.clip(db_tmp["sum_requested_prbs"], a_min=0, a_max=None) / db_tmp["sum_granted_prbs"])})
    #     return db_tmp.join(dict_add)
    #
    #
    # def get_bottom_rows_from_file(filename, chunksize=1000):  # this is a version that works best for small files
    #     db_tmp = pd.read_csv(filename)
    #     db_tmp = db_tmp.tail(n=chunksize).reset_index(drop=True)
    #     dict_add = pd.DataFrame.from_dict({"ratio_req_granted": np.nan_to_num(
    #         np.clip(db_tmp["sum_requested_prbs"], a_min=0, a_max=None) / db_tmp["sum_granted_prbs"])})
    #     return db_tmp.join(dict_add)
    #
    #
    # def bottom_batches_from_folders(main_folder, wildcard, batch_size=16, n_batches=1, is_large=False):
    #     dataset = []
    #     chunk_size = batch_size * n_batches
    #     for filename in glob.glob(main_folder + wildcard):
    #         if not is_large:
    #             db_tmp = get_bottom_rows_from_file(filename=filename, chunksize=chunk_size)
    #         else:
    #             db_tmp = get_bottom_rows_from_file_large(filename=filename, chunksize=chunk_size)
    #         dataset.append(db_tmp)
    #     return pd.concat(dataset, axis=0, ignore_index=True)

    # # TODO: extract by slice_id
    # def bottom_batches_from_folders_per_slice(main_folder, wildcard, batch_size=16, n_batches=1, slice_id=None):
    #     dataset = []
    #     chunk_size = batch_size * n_batches
    #     for filename in glob.glob(main_folder + wildcard):
    #         db_tmp = get_bottom_rows_from_file(filename=filename, chunksize=chunk_size)
    #         dataset.append(db_tmp)
    #     return pd.concat(dataset, axis=0, ignore_index=True)
