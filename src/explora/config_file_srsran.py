###########################
#### DATASET METRICS ######
###########################

# folder where data is stored
main_folder = '../colosseum-oran-coloran-dataset/rome_static_medium'
# main_folder = '/home/microway/mpolese/cellular_dataset_colosseum/datasets/slice_traffic'
wildcard_match = '/*/*/*/*/*/*_metrics.csv'

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

scheduling_combos = [[0,0,0],
        [0,0,1],
        [0,0,2],
        [0,1,0],
        [0,1,1],
        [0,1,2],
        [0,2,0],
        [0,2,1],
        [0,2,2],
        [1,0,0],
        [1,0,1],
        [1,0,2],
        [1,1,0],
        [1,1,1],
        [1,1,2],
        [1,2,0],
        [1,2,1],
        [1,2,2],
        [2,0,0],
        [2,0,1],
        [2,0,2],
        [2,1,0],
        [2,1,1],
        [2,1,2],
        [2,2,0],
        [2,2,1],
        [2,2,2]]

feasible_prb_allocation_all = [[30, 9, 11],
                       [30, 15, 5],
                       [36, 9, 5],
                       [24, 21, 5],
                       [24, 15, 11],
                       [18, 15, 17],
                       [18, 9, 23],
                       [18, 21, 11],
                       [12, 27, 11],
                       [12, 15, 23],
                       [12, 9, 29],
                       [6, 27, 17],
                       [6, 39, 5],
                       [6, 15, 29],
                       [6, 9, 35],
                       [36, 3, 11]
                       ]

feasible_prb_allocation_indexes = [0, 1, 2, 3, 4, 5, 6, 9, 12, 14, 15]
feasible_prb_allocation = [feasible_prb_allocation_all[int(x)] for x in feasible_prb_allocation_indexes]

###########################
#### PARAMS  METRICS ######
###########################

is_load_dataset_from_csv = False  # if False, it loads the file below. In the automated case, this will never happen as this will always load data from the CSV files
is_save_dataset_to_pickle = False
dataset_complete_name = '../dataset_complete_new_no_buffer_scaling.pkl'
remove_zero_req_prb_entries = False
remove_zero_throughput_entries = True
remove_zero_cqi_entries = True
remove_zero_mcs_entries = True
simulate_zero_entries = False
add_prb_ratio = False
verbose = False


###########################
#### TRAINING METRICS #####
###########################

action_dtype = 'int32'
n_entries_per_observation = 10  # must match the input size of the autoencoder if using one