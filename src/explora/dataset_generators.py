import abc

from pandas import DataFrame, concat

import srslte_utils as sr
import numpy as np
from intent_constants import IntentConstants


class DatasetGenerator(abc.ABC):
    def __init__(self,
                 dataset=None,
                 observation_format=None):
        self.dataset = dataset
        self.observation_format = observation_format
        pass

    @abc.abstractmethod
    def get_next_observation(self,
                             action_idx = None):
        pass

    def set_dataset(self,
                    dataset=None):
        self.dataset = dataset

    def set_observation_format(self,
                               observation_format=None):
        self.observation_format = observation_format


class SRSDatasetGenerator(DatasetGenerator):
    def __init__(self,
                 observation_format=None,
                 slice_ids=None,
                 KPIs_to_extract=None,
                 non_KPIs_to_extract=None,
                 config_file=None,
                 action_profile=None,
                 simulate_zero_entries=False
                 ):

        # Add actions to the list of things to extract from the CSV file. If you don't extract them, you won't be able to extract data in the offline training
        if KPIs_to_extract is not None:
            self.columns_to_extract = KPIs_to_extract.copy()
            if non_KPIs_to_extract is not None:
                self.columns_to_extract.extend(non_KPIs_to_extract)

        # Load entire dataset (this extracts the KPIs of interest only)
        dataset = self.load_srs_dataset(config_file=config_file,
                                        columns_to_extract=self.columns_to_extract)

        self.slice_ids = slice_ids
        if slice_ids:
            self.n_slices = len(self.slice_ids)
            self.n_entries_per_obs = [observation_format.input_shape[i][0] for i in range(len(slice_ids))] # this is just for srsRAN where the rows represent the entries

        if action_profile:
            self.action_profile = action_profile

        self.simulate_zero_entries = simulate_zero_entries

        # The dataset we pass here includes only data with desired slice_ids and metrics we need
        super().__init__(dataset=sr.extract_dataset_for_training_v2(dataset=dataset,
                                                                    slice_ids=self.slice_ids),
                         observation_format=observation_format)

    def load_srs_dataset(self,
                         columns_to_extract: list,
                         config_file=None):
        if config_file is not None:
            if config_file.is_load_dataset_from_csv:
                dataset = sr.entire_dataset_from_folder(main_folder=config_file.main_folder,
                                                        wildcard=config_file.wildcard_match,
                                                        col_names=config_file.metrics_list,
                                                        selected_col_names=columns_to_extract,
                                                        remove_zero_throughput_entries=config_file.remove_zero_throughput_entries,
                                                        remove_zero_req_prb_entries=config_file.remove_zero_req_prb_entries,
                                                        remove_zero_cqi_entries=config_file.remove_zero_cqi_entries,
                                                        remove_zero_mcs_entries=config_file.remove_zero_mcs_entries,
                                                        add_prb_ratio=config_file.add_prb_ratio
                                                        )
                if config_file.is_save_dataset_to_pickle:
                    dataset.to_pickle(config_file.dataset_complete_name)
            else:
                import pandas as pd
                dataset = pd.read_pickle(config_file.dataset_complete_name)

                if [dataset.keys().__contains__(columns_to_extract[i]) for i in range(len(columns_to_extract))].__contains__(False):
                    print("Dataset does not contain the required metrics to extract, either load from CSV files or re-generate the static dataset with the desired KPIs.")
                    exit(-1)


            return dataset
        else:
            print('Error in loading the dataset')
            exit(-1)

    # TODO: this
    def get_next_observation(self,
                             action_idx = None):

        # action_dict =  {'ran_slicing': {'action': [20 10 20], 'metric': 'slice_prb'}, etc... } look at Intent.ActionTypes() for a detailed definition
        action_dict = self.action_profile.get_selected_action_from_idx(action_idx=action_idx)

        # Here we are converting the action in a way we can extract something from the DataFrame (i.e., the dataset)
        slice_prb, scheduling_profile = None, None   # initilaizing this in case actions are not in here

        # TODO: this will have to be updated in case we have more actions
        for k in action_dict.keys():
            if k == IntentConstants.ActionType.SCHEDULING['name_human']:
                scheduling_profile = action_dict[k]['action']
            elif k == IntentConstants.ActionType.RAN_SLICING['name_human']:
                slice_prb = action_dict[k]['action']

        # get data for observation
        obs = sr.extract_observation_from_dataset(dataset=self.dataset,
                                                  slice_ids=self.slice_ids,
                                                  slice_prb=slice_prb,
                                                  metrics_export=self.observation_format.kpi_names,
                                                  scheduling_policy=scheduling_profile,
                                                  n_sample=[self.n_entries_per_obs[i] for i in
                                                            range(self.n_slices)])

        if self.simulate_zero_entries:
            # TODO: put zero entries
            # compute amount of data to extract to get an observation
            n_rem = [np.random.randint(0, self.n_entries_per_obs[i] - 1) for i in range(self.n_slices)]
            # create empty entries
            zero_df = [DataFrame(np.zeros((n_rem[i], len(self.observation_format.kpi_names[i]))),
                                 columns=self.observation_format.kpi_names[i]) for i in range(self.n_slices)]

            # remove bottom entries
            for i in range(self.n_slices):
                if len(zero_df[i]):
                    obs[i] = obs[i][0: (self.n_entries_per_obs[i] - n_rem[i])]
                    obs[i] = concat([obs[i], zero_df[i]], ignore_index=True).reset_index(drop=True)

            return obs
        else:
            return [o.reset_index(drop=True) for o in obs]

