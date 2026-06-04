from dataset_generators import SRSDatasetGenerator
import config_file_srsran_2_slices as cfg

if __name__ == '__main__':

    columns_to_extract = [
        "slice_id",
        "slice_prb",
        "scheduling_policy",
        "dl_mcs",
        "dl_n_samples",
        "dl_buffer [bytes]",
        "tx_brate downlink [Mbps]",
        "tx_pkts downlink",
        "dl_cqi",
        "ul_mcs",
        "ul_n_samples",
        "ul_buffer [bytes]",
        "rx_brate uplink [Mbps]",
        "rx_pkts uplink",
        "rx_errors uplink (%)",
        "ul_sinr",
        "sum_requested_prbs",
        "sum_granted_prbs"]

    cfg.is_load_dataset_from_csv = True
    cfg.is_save_dataset_to_pickle = True
    cfg.remove_zero_req_prb_entries = False
    cfg.remove_zero_throughput_entries = True
    cfg.remove_zero_cqi_entries = True
    cfg.remove_zero_mcs_entries = True
    cfg.simulate_zero_entries = False
    cfg.add_prb_ratio = True
    cfg.verbose = True

    cfg.dataset_complete_name = '../dataset_new_explora_k3.pkl'

    print("Dataset generation started")

    dataset_generator = SRSDatasetGenerator(config_file=cfg,
                                            KPIs_to_extract=columns_to_extract)

    print("Dataset generated and saved to ", cfg.dataset_complete_name)