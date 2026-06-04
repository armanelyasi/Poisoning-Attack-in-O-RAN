###########################
#### DATASET METRICS ######
###########################

is_load_dataset_from_csv = False  # if False, it loads the file below. In the automated case, this will never happen as this will always load data from the CSV files
is_save_dataset_to_pickle = False
dataset_complete_name = '/share/explora-dataset-pkl/dataset_new_explora_k12.pkl'
remove_zero_req_prb_entries = False
remove_zero_throughput_entries = True
remove_zero_cqi_entries = True
remove_zero_mcs_entries = True
simulate_zero_entries = False
add_prb_ratio = False
verbose = False

# folder where data is stored
main_folder = '/media/salvo/Extreme SSD/explora-dataset/cluster_3'
wildcard_match = '/*/*/*/*/*/metrics/csv/*_metrics.csv'

# all KPIs stored in the dataset (in this case CSV-formatted data and these are the names of the columns
metrics_list = ["Timestamp",
                "num_ues",
                "IMSI",
                "RNTI",
                "empty_1",
                "slicing_enabled",
                "slice_id",
                "slice_prb",
                "power_multiplier",
                "scheduling_policy",
                "empty_2",
                "dl_mcs",
                "dl_n_samples",
                "dl_buffer [bytes]",
                "tx_brate downlink [Mbps]",
                "tx_pkts downlink",
                "tx_errors downlink (%)",
                "dl_cqi",
                "empty_3",
                "ul_mcs",
                "ul_n_samples",
                "ul_buffer [bytes]",
                "rx_brate uplink [Mbps]",
                "rx_pkts uplink",
                "rx_errors uplink (%)",
                "ul_rssi",
                "ul_sinr",
                "phr",
                "empty_4",
                "sum_requested_prbs",
                "sum_granted_prbs",
                "empty_5",
                "dl_pmi",
                "dl_ri",
                "ul_n",
                "ul_turbo_iters"]

non_KPI_columns_to_extract = ["slice_id",
                      "slice_prb",
                      "scheduling_policy"]


###########################
#### PARAMS  METRICS ######
###########################

scheduling_combos = [
        [0,0],
        [0,2],
        [2,0],
        [2,2]]

feasible_prb_allocation_all = [[9,41],
                       [21, 29],
                       [30, 20],
                       [39, 11]
                       ]

feasible_prb_allocation_indexes = [0, 1, 2, 3]
feasible_prb_allocation = [feasible_prb_allocation_all[int(x)] for x in feasible_prb_allocation_indexes]

###########################
#### TRAINING METRICS #####
###########################

action_dtype = 'int32'
n_entries_per_observation = 10  # must match the input size of the autoencoder if using one
