import copy

import numpy as np
import torch
import h5py
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
from PIL import Image
from robomimic.utils.dataset import SequenceDataset
from torch.utils.data import Dataset


import os
import h5py
import numpy as np
from copy import deepcopy
from contextlib import contextmanager

import torch.utils.data

import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.log_utils as LogUtils


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            hdf5_path,
            obs_keys,
            dataset_keys,
            frame_stack=1,
            seq_length=1,
            pad_frame_stack=True,
            pad_seq_length=True,
            get_pad_mask=False,
            goal_mode=None,
            hdf5_cache_mode=None,
            hdf5_use_swmr=True,
            hdf5_normalize_obs=False,
            filter_by_attribute=None,
            load_next_obs=True,
    ):
        """
        Dataset class for fetching sequences of experience.
        Length of the fetched sequence is equal to (@frame_stack - 1 + @seq_length)

        Args:
            hdf5_path (str): path to hdf5

            obs_keys (tuple, list): keys to observation items (image, object, etc) to be fetched from the dataset

            dataset_keys (tuple, list): keys to dataset items (actions, rewards, etc) to be fetched from the dataset

            frame_stack (int): numbers of stacked frames to fetch. Defaults to 1 (single frame).

            seq_length (int): length of sequences to sample. Defaults to 1 (single frame).

            pad_frame_stack (int): whether to pad sequence for frame stacking at the beginning of a demo. This
                ensures that partial frame stacks are observed, such as (s_0, s_0, s_0, s_1). Otherwise, the
                first frame stacked observation would be (s_0, s_1, s_2, s_3).

            pad_seq_length (int): whether to pad sequence for sequence fetching at the end of a demo. This
                ensures that partial sequences at the end of a demonstration are observed, such as
                (s_{T-1}, s_{T}, s_{T}, s_{T}). Otherwise, the last sequence provided would be
                (s_{T-3}, s_{T-2}, s_{T-1}, s_{T}).

            get_pad_mask (bool): if True, also provide padding masks as part of the batch. This can be
                useful for masking loss functions on padded parts of the data.

            goal_mode (str): either "last" or None. Defaults to None, which is to not fetch goals

            hdf5_cache_mode (str): one of ["all", "low_dim", or None]. Set to "all" to cache entire hdf5
                in memory - this is by far the fastest for data loading. Set to "low_dim" to cache all
                non-image data. Set to None to use no caching - in this case, every batch sample is
                retrieved via file i/o. You should almost never set this to None, even for large
                image datasets.

            hdf5_use_swmr (bool): whether to use swmr feature when opening the hdf5 file. This ensures
                that multiple Dataset instances can all access the same hdf5 file without problems.

            hdf5_normalize_obs (bool): if True, normalize observations by computing the mean observation
                and std of each observation (in each dimension and modality), and normalizing to unit
                mean and variance in each dimension.

            filter_by_attribute (str): if provided, use the provided filter key to look up a subset of
                demonstrations to load

            load_next_obs (bool): whether to load next_obs from the dataset
        """
        super(SequenceDataset, self).__init__()

        self.hdf5_path = os.path.expanduser(hdf5_path)
        self.hdf5_use_swmr = hdf5_use_swmr
        self.hdf5_normalize_obs = hdf5_normalize_obs
        self._hdf5_file = None

        assert hdf5_cache_mode in ["all", "low_dim", None]
        self.hdf5_cache_mode = hdf5_cache_mode

        self.load_next_obs = load_next_obs
        self.filter_by_attribute = filter_by_attribute

        # get all keys that needs to be fetched
        self.obs_keys = tuple(obs_keys)
        self.dataset_keys = tuple(dataset_keys)

        self.n_frame_stack = frame_stack
        assert self.n_frame_stack >= 1

        self.seq_length = seq_length
        assert self.seq_length >= 1

        self.goal_mode = goal_mode
        if self.goal_mode is not None:
            assert self.goal_mode in ["last"]
        if not self.load_next_obs:
            assert self.goal_mode != "last"  # we use last next_obs as goal

        self.pad_seq_length = pad_seq_length
        self.pad_frame_stack = pad_frame_stack
        self.get_pad_mask = get_pad_mask

        self.load_demo_info(filter_by_attribute=self.filter_by_attribute)

        # maybe prepare for observation normalization
        self.obs_normalization_stats = None
        if self.hdf5_normalize_obs:
            self.obs_normalization_stats = self.normalize_obs()

        # maybe store dataset in memory for fast access
        if self.hdf5_cache_mode in ["all", "low_dim"]:
            obs_keys_in_memory = self.obs_keys
            if self.hdf5_cache_mode == "low_dim":
                # only store low-dim observations
                obs_keys_in_memory = []
                for k in self.obs_keys:
                    if ObsUtils.key_is_obs_modality(k, "low_dim"):
                        obs_keys_in_memory.append(k)
            self.obs_keys_in_memory = obs_keys_in_memory

            self.hdf5_cache = self.load_dataset_in_memory(
                demo_list=self.demos,
                hdf5_file=self.hdf5_file,
                obs_keys=self.obs_keys_in_memory,
                dataset_keys=self.dataset_keys,
                load_next_obs=self.load_next_obs
            )

            if self.hdf5_cache_mode == "all":
                # cache getitem calls for even more speedup. We don't do this for
                # "low-dim" since image observations require calls to getitem anyways.
                print("SequenceDataset: caching get_item calls...")
                self.getitem_cache = [self.get_item(i) for i in LogUtils.custom_tqdm(range(len(self)))]

                # don't need the previous cache anymore
                del self.hdf5_cache
                self.hdf5_cache = None
        else:
            self.hdf5_cache = None

        self.close_and_delete_hdf5_handle()

    def load_demo_info(self, filter_by_attribute=None, demos=None):
        """
        Args:
            filter_by_attribute (str): if provided, use the provided filter key
                to select a subset of demonstration trajectories to load

            demos (list): list of demonstration keys to load from the hdf5 file. If
                omitted, all demos in the file (or under the @filter_by_attribute
                filter key) are used.
        """
        # filter demo trajectory by mask
        if demos is not None:
            self.demos = demos
        elif filter_by_attribute is not None:
            self.demos = [elem.decode("utf-8") for elem in np.array(self.hdf5_file["mask/{}".format(filter_by_attribute)][:])]
        else:
            self.demos = list(self.hdf5_file["data"].keys())

        # sort demo keys
        inds = np.argsort([int(elem[5:]) for elem in self.demos])
        self.demos = [self.demos[i] for i in inds]

        self.n_demos = len(self.demos)

        # keep internal index maps to know which transitions belong to which demos
        self._index_to_demo_id = dict()  # maps every index to a demo id
        self._demo_id_to_start_indices = dict()  # gives start index per demo id
        self._demo_id_to_demo_length = dict()

        # determine index mapping
        self.total_num_sequences = 0
        for ep in self.demos:
            demo_length = self.hdf5_file["data/{}".format(ep)].attrs["num_samples"]
            self._demo_id_to_start_indices[ep] = self.total_num_sequences
            self._demo_id_to_demo_length[ep] = demo_length

            num_sequences = demo_length
            # determine actual number of sequences taking into account whether to pad for frame_stack and seq_length
            if not self.pad_frame_stack:
                num_sequences -= (self.n_frame_stack - 1)
            if not self.pad_seq_length:
                num_sequences -= (self.seq_length - 1)

            if self.pad_seq_length:
                assert demo_length >= 1  # sequence needs to have at least one sample
                num_sequences = max(num_sequences, 1)
            else:
                assert num_sequences >= 1  # assume demo_length >= (self.n_frame_stack - 1 + self.seq_length)

            for _ in range(num_sequences):
                self._index_to_demo_id[self.total_num_sequences] = ep
                self.total_num_sequences += 1

    @property
    def hdf5_file(self):
        """
        This property allows for a lazy hdf5 file open.
        """
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, 'r', swmr=self.hdf5_use_swmr, libver='latest')
        return self._hdf5_file

    def close_and_delete_hdf5_handle(self):
        """
        Maybe close the file handle.
        """
        if self._hdf5_file is not None:
            self._hdf5_file.close()
        self._hdf5_file = None

    @contextmanager
    def hdf5_file_opened(self):
        """
        Convenient context manager to open the file on entering the scope
        and then close it on leaving.
        """
        should_close = self._hdf5_file is None
        yield self.hdf5_file
        if should_close:
            self.close_and_delete_hdf5_handle()

    def __del__(self):
        self.close_and_delete_hdf5_handle()

    def __repr__(self):
        """
        Pretty print the class and important attributes on a call to `print`.
        """
        msg = str(self.__class__.__name__)
        msg += " (\n\tpath={}\n\tobs_keys={}\n\tseq_length={}\n\tfilter_key={}\n\tframe_stack={}\n"
        msg += "\tpad_seq_length={}\n\tpad_frame_stack={}\n\tgoal_mode={}\n"
        msg += "\tcache_mode={}\n"
        msg += "\tnum_demos={}\n\tnum_sequences={}\n)"
        filter_key_str = self.filter_by_attribute if self.filter_by_attribute is not None else "none"
        goal_mode_str = self.goal_mode if self.goal_mode is not None else "none"
        cache_mode_str = self.hdf5_cache_mode if self.hdf5_cache_mode is not None else "none"
        msg = msg.format(self.hdf5_path, self.obs_keys, self.seq_length, filter_key_str, self.n_frame_stack,
                         self.pad_seq_length, self.pad_frame_stack, goal_mode_str, cache_mode_str,
                         self.n_demos, self.total_num_sequences)
        return msg

    def __len__(self):
        """
        Ensure that the torch dataloader will do a complete pass through all sequences in
        the dataset before starting a new iteration.
        """
        return self.total_num_sequences

    def load_dataset_in_memory(self, demo_list, hdf5_file, obs_keys, dataset_keys, load_next_obs):
        """
        Loads the hdf5 dataset into memory, preserving the structure of the file. Note that this
        differs from `self.getitem_cache`, which, if active, actually caches the outputs of the
        `getitem` operation.

        Args:
            demo_list (list): list of demo keys, e.g., 'demo_0'
            hdf5_file (h5py.File): file handle to the hdf5 dataset.
            obs_keys (list, tuple): observation keys to fetch, e.g., 'images'
            dataset_keys (list, tuple): dataset keys to fetch, e.g., 'actions'
            load_next_obs (bool): whether to load next_obs from the dataset

        Returns:
            all_data (dict): dictionary of loaded data.
        """
        all_data = dict()
        print("SequenceDataset: loading dataset into memory...")
        for ep in LogUtils.custom_tqdm(demo_list):
            all_data[ep] = {}
            all_data[ep]["attrs"] = {}
            all_data[ep]["attrs"]["num_samples"] = hdf5_file["data/{}".format(ep)].attrs["num_samples"]
            # get obs
            all_data[ep]["obs"] = {k: hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype('float32') for k in obs_keys}
            if load_next_obs:
                all_data[ep]["next_obs"] = {k: hdf5_file["data/{}/next_obs/{}".format(ep, k)][()].astype('float32') for k in obs_keys}
            # get other dataset keys
            for k in dataset_keys:
                if k in hdf5_file["data/{}".format(ep)]:
                    all_data[ep][k] = hdf5_file["data/{}/{}".format(ep, k)][()].astype('float32')
                else:
                    all_data[ep][k] = np.zeros((all_data[ep]["attrs"]["num_samples"], 1), dtype=np.float32)

            if "model_file" in hdf5_file["data/{}".format(ep)].attrs:
                all_data[ep]["attrs"]["model_file"] = hdf5_file["data/{}".format(ep)].attrs["model_file"]

        return all_data

    def normalize_obs(self):
        """
        Computes a dataset-wide mean and standard deviation for the observations
        (per dimension and per obs key) and returns it.
        """

        def _compute_traj_stats(traj_obs_dict):
            """
            Helper function to compute statistics over a single trajectory of observations.
            """
            traj_stats = {k: {} for k in traj_obs_dict}
            for k in traj_obs_dict:
                traj_stats[k]["n"] = traj_obs_dict[k].shape[0]
                traj_stats[k]["mean"] = traj_obs_dict[k].mean(axis=0, keepdims=True)  # [1, ...]
                traj_stats[k]["sqdiff"] = ((traj_obs_dict[k] - traj_stats[k]["mean"]) ** 2).sum(axis=0, keepdims=True)  # [1, ...]
            return traj_stats

        def _aggregate_traj_stats(traj_stats_a, traj_stats_b):
            """
            Helper function to aggregate trajectory statistics.
            See https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
            for more information.
            """
            merged_stats = {}
            for k in traj_stats_a:
                n_a, avg_a, M2_a = traj_stats_a[k]["n"], traj_stats_a[k]["mean"], traj_stats_a[k]["sqdiff"]
                n_b, avg_b, M2_b = traj_stats_b[k]["n"], traj_stats_b[k]["mean"], traj_stats_b[k]["sqdiff"]
                n = n_a + n_b
                mean = (n_a * avg_a + n_b * avg_b) / n
                delta = (avg_b - avg_a)
                M2 = M2_a + M2_b + (delta ** 2) * (n_a * n_b) / n
                merged_stats[k] = dict(n=n, mean=mean, sqdiff=M2)
            return merged_stats

        # Run through all trajectories. For each one, compute minimal observation statistics, and then aggregate
        # with the previous statistics.
        ep = self.demos[0]
        obs_traj = {k: self.hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype('float32') for k in self.obs_keys}
        obs_traj = ObsUtils.process_obs_dict(obs_traj)
        merged_stats = _compute_traj_stats(obs_traj)
        print("SequenceDataset: normalizing observations...")
        for ep in LogUtils.custom_tqdm(self.demos[1:]):
            obs_traj = {k: self.hdf5_file["data/{}/obs/{}".format(ep, k)][()].astype('float32') for k in self.obs_keys}
            obs_traj = ObsUtils.process_obs_dict(obs_traj)
            traj_stats = _compute_traj_stats(obs_traj)
            merged_stats = _aggregate_traj_stats(merged_stats, traj_stats)

        obs_normalization_stats = {k: {} for k in merged_stats}
        for k in merged_stats:
            # note we add a small tolerance of 1e-3 for std
            obs_normalization_stats[k]["mean"] = merged_stats[k]["mean"]
            obs_normalization_stats[k]["std"] = np.sqrt(merged_stats[k]["sqdiff"] / merged_stats[k]["n"]) + 1e-3
        return obs_normalization_stats

    def get_obs_normalization_stats(self):
        """
        Returns dictionary of mean and std for each observation key if using
        observation normalization, otherwise None.

        Returns:
            obs_normalization_stats (dict): a dictionary for observation
                normalization. This maps observation keys to dicts
                with a "mean" and "std" of shape (1, ...) where ... is the default
                shape for the observation.
        """
        assert self.hdf5_normalize_obs, "not using observation normalization!"
        return deepcopy(self.obs_normalization_stats)

    def get_dataset_for_ep(self, ep, key):
        """
        Helper utility to get a dataset for a specific demonstration.
        Takes into account whether the dataset has been loaded into memory.
        """

        # check if this key should be in memory
        key_should_be_in_memory = (self.hdf5_cache_mode in ["all", "low_dim"])
        if key_should_be_in_memory:
            # if key is an observation, it may not be in memory
            if '/' in key:
                key1, key2 = key.split('/')
                assert (key1 in ['obs', 'next_obs'])
                if key2 not in self.obs_keys_in_memory:
                    key_should_be_in_memory = False

        if key_should_be_in_memory:
            # read cache
            if '/' in key:
                key1, key2 = key.split('/')
                assert (key1 in ['obs', 'next_obs'])
                ret = self.hdf5_cache[ep][key1][key2]
            else:
                ret = self.hdf5_cache[ep][key]
        else:
            # read from file
            hd5key = "data/{}/{}".format(ep, key)
            ret = self.hdf5_file[hd5key]
        return ret

    def __getitem__(self, index):
        """
        Fetch dataset sequence @index (inferred through internal index map), using the getitem_cache if available.
        """
        if self.hdf5_cache_mode == "all":
            return self.getitem_cache[index]
        return self.get_item(index)

    def get_item(self, index):
        """
        Main implementation of getitem when not using cache.
        """

        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]
        demo_length = self._demo_id_to_demo_length[demo_id]

        # start at offset index if not padding for frame stacking
        demo_index_offset = 0 if self.pad_frame_stack else (self.n_frame_stack - 1)
        index_in_demo = index - demo_start_index + demo_index_offset

        # end at offset index if not padding for seq length
        demo_length_offset = 0 if self.pad_seq_length else (self.seq_length - 1)
        end_index_in_demo = demo_length - demo_length_offset

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.dataset_keys,
            seq_length=self.seq_length
        )

        # determine goal index
        goal_index = None
        if self.goal_mode == "last":
            goal_index = end_index_in_demo - 1

        meta["obs"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=self.obs_keys,
            num_frames_to_stack=self.n_frame_stack - 1,
            # seq_length=self.seq_length,
            seq_length=1,
            prefix="obs"
        )
        if self.hdf5_normalize_obs:
            meta["obs"] = ObsUtils.normalize_obs(meta["obs"], obs_normalization_stats=self.obs_normalization_stats)

        if self.load_next_obs:
            meta["next_obs"] = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=index_in_demo,
                keys=self.obs_keys,
                num_frames_to_stack=self.n_frame_stack - 1,
                seq_length=self.seq_length,
                prefix="next_obs"
            )
            if self.hdf5_normalize_obs:
                meta["next_obs"] = ObsUtils.normalize_obs(meta["next_obs"], obs_normalization_stats=self.obs_normalization_stats)

        if goal_index is not None:
            goal = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=goal_index,
                keys=self.obs_keys,
                num_frames_to_stack=0,
                seq_length=1,
                prefix="next_obs",
            )
            if self.hdf5_normalize_obs:
                goal = ObsUtils.normalize_obs(goal, obs_normalization_stats=self.obs_normalization_stats)
            meta["goal_obs"] = {k: goal[k][0] for k in goal}  # remove sequence dimension for goal

        return meta

    def get_sequence_from_demo(self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1):
        """
        Extract a (sub)sequence of data items from a demo given the @keys of the items.

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range

        Returns:
            a dictionary of extracted items.
        """
        assert num_frames_to_stack >= 0
        assert seq_length >= 1

        demo_length = self._demo_id_to_demo_length[demo_id]
        assert index_in_demo < demo_length

        # determine begin and end of sequence
        seq_begin_index = max(0, index_in_demo - num_frames_to_stack)
        seq_end_index = min(demo_length, index_in_demo + seq_length)

        # determine sequence padding
        seq_begin_pad = max(0, num_frames_to_stack - index_in_demo)  # pad for frame stacking
        seq_end_pad = max(0, index_in_demo + seq_length - demo_length)  # pad for sequence length

        # make sure we are not padding if specified.
        if not self.pad_frame_stack:
            assert seq_begin_pad == 0
        if not self.pad_seq_length:
            assert seq_end_pad == 0

        # fetch observation from the dataset file
        seq = dict()
        for k in keys:
            data = self.get_dataset_for_ep(demo_id, k)
            seq[k] = data[seq_begin_index: seq_end_index].astype("float32")

        seq = TensorUtils.pad_sequence(seq, padding=(seq_begin_pad, seq_end_pad), pad_same=True)
        pad_mask = np.array([0] * seq_begin_pad + [1] * (seq_end_index - seq_begin_index) + [0] * seq_end_pad)
        pad_mask = pad_mask[:, None].astype(bool)

        return seq, pad_mask

    def get_obs_sequence_from_demo(self, demo_id, index_in_demo, keys, num_frames_to_stack=0, seq_length=1, prefix="obs"):
        """
        Extract a (sub)sequence of observation items from a demo given the @keys of the items.

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            num_frames_to_stack (int): numbers of frame to stack. Seq gets prepended with repeated items if out of range
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range
            prefix (str): one of "obs", "next_obs"

        Returns:
            a dictionary of extracted items.
        """
        obs, pad_mask = self.get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=tuple('{}/{}'.format(prefix, k) for k in keys),
            num_frames_to_stack=num_frames_to_stack,
            seq_length=seq_length,
        )
        obs = {k.split('/')[1]: obs[k] for k in obs}  # strip the prefix
        if self.get_pad_mask:
            obs["pad_mask"] = pad_mask

        # prepare image observations from dataset
        return ObsUtils.process_obs_dict(obs)

    def get_dataset_sequence_from_demo(self, demo_id, index_in_demo, keys, seq_length=1):
        """
        Extract a (sub)sequence of dataset items from a demo given the @keys of the items (e.g., states, actions).

        Args:
            demo_id (str): id of the demo, e.g., demo_0
            index_in_demo (int): beginning index of the sequence wrt the demo
            keys (tuple): list of keys to extract
            seq_length (int): sequence length to extract. Seq gets post-pended with repeated items if out of range

        Returns:
            a dictionary of extracted items.
        """
        data, pad_mask = self.get_sequence_from_demo(
            demo_id,
            index_in_demo=index_in_demo,
            keys=keys,
            num_frames_to_stack=0,  # don't frame stack for meta keys
            seq_length=seq_length,
        )
        if self.get_pad_mask:
            data["pad_mask"] = pad_mask
        return data

    def get_trajectory_at_index(self, index):
        """
        Method provided as a utility to get an entire trajectory, given
        the corresponding @index.
        """
        demo_id = self.demos[index]
        demo_length = self._demo_id_to_demo_length[demo_id]

        meta = self.get_dataset_sequence_from_demo(
            demo_id,
            index_in_demo=0,
            keys=self.dataset_keys,
            seq_length=demo_length
        )
        meta["obs"] = self.get_obs_sequence_from_demo(
            demo_id,
            index_in_demo=0,
            keys=self.obs_keys,
            seq_length=demo_length
        )
        if self.load_next_obs:
            meta["next_obs"] = self.get_obs_sequence_from_demo(
                demo_id,
                index_in_demo=0,
                keys=self.obs_keys,
                seq_length=demo_length,
                prefix="next_obs"
            )

        meta["ep"] = demo_id
        return meta

    def get_dataset_sampler(self):
        """
        Return instance of torch.utils.data.Sampler or None. Allows
        for dataset to define custom sampling logic, such as
        re-weighting the probability of samples being drawn.
        See the `train` function in scripts/train.py, and torch
        `DataLoader` documentation, for more info.
        """
        return None


"""
    Helper function from Robomimic to read hdf5 demonstrations into sequence dataset

    ISSUE: robomimic's SequenceDataset has two properties: seq_len and frame_stack,
    we should in principle use seq_len, but the paddings of the two are different.
    So that's why we currently use frame_stack instead of seq_len.
"""


def get_dataset(
    dataset_path,
    obs_modality,
    initialize_obs_utils=True,
    seq_len=1,
    frame_stack=1,
    filter_key=None,
    hdf5_cache_mode="low_dim",
    *args,
    **kwargs
):

    if initialize_obs_utils:
        ObsUtils.initialize_obs_utils_with_obs_specs({"obs": obs_modality})

    all_obs_keys = []
    for modality_name, modality_list in obs_modality.items():
        all_obs_keys += modality_list
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path, all_obs_keys=all_obs_keys, verbose=False
    )

    seq_len = seq_len 
    filter_key = filter_key
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=shape_meta["all_obs_keys"],
        dataset_keys=["actions"],
        load_next_obs=False,
        frame_stack=frame_stack,
        seq_length=seq_len,  # length-10 temporal sequences
        pad_frame_stack=True,
        pad_seq_length=True,  # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode=hdf5_cache_mode,  # cache dataset in memory to avoid repeated file i/o
        hdf5_use_swmr=False,
        hdf5_normalize_obs=None,
        filter_by_attribute=filter_key,  # can optionally provide a filter key here
        # demos=kwargs['demos'],
    )
    return dataset, shape_meta


class SequenceVLDataset(Dataset):
    def __init__(self, sequence_dataset, task_emb):
        self.sequence_dataset = sequence_dataset
        self.task_emb = task_emb
        self.n_demos = self.sequence_dataset.n_demos
        self.total_num_sequences = self.sequence_dataset.total_num_sequences

    def __len__(self):
        return len(self.sequence_dataset)

    def __getitem__(self, idx):
        return_dict = self.sequence_dataset.__getitem__(idx)
        return_dict["task_emb"] = self.task_emb
        return return_dict


class GroupedTaskDataset(Dataset):
    def __init__(self, sequence_datasets, task_embs):
        self.sequence_datasets = sequence_datasets
        self.task_embs = task_embs
        self.group_size = len(sequence_datasets)
        self.n_demos = sum([x.n_demos for x in self.sequence_datasets])
        self.total_num_sequences = sum(
            [x.total_num_sequences for x in self.sequence_datasets]
        )
        self.lengths = [len(x) for x in self.sequence_datasets]
        self.task_group_size = len(self.sequence_datasets)

        # create a map that maps the current idx of dataloader to original task data idx
        # imagine we have task 1,2,3, with sizes 3,5,4, then the idx looks like
        # task-1  task-2  task-3
        #   0       1       2
        #   3       4       5
        #   6       7       8
        #           9       10
        #           11
        # by doing so, when we concat the dataset, every task will have equal number of demos
        self.map_dict = {}
        sizes = np.array(self.lengths)
        row = 0
        col = 0
        for i in range(sum(sizes)):
            while sizes[col] == 0:
                col = col + 1
                if col >= self.task_group_size:
                    col -= self.task_group_size
                    row += 1
            self.map_dict[i] = (row, col)
            sizes[col] -= 1
            col += 1
            if col >= self.task_group_size:
                col -= self.task_group_size
                row += 1
        self.n_total = sum(self.lengths)

    def __len__(self):
        return self.n_total

    def __get_original_task_idx(self, idx):
        return self.map_dict[idx]

    def __getitem__(self, idx):
        oi, oti = self.__get_original_task_idx(idx)
        return_dict = self.sequence_datasets[oti].__getitem__(oi)
        return_dict["task_emb"] = self.task_embs[oti]
        return return_dict


class TruncatedSequenceDataset(Dataset):
    def __init__(self, sequence_dataset, buffer_size):
        self.sequence_dataset = sequence_dataset
        self.buffer_size = buffer_size

    def __len__(self):
        return self.buffer_size

    def __getitem__(self, idx):
        return self.sequence_dataset.__getitem__(idx)
    
new_task_demo_num = 50
old_task_demo_num = 10

## Skill Learning Dataset
from libero.lifelong.algos.lotus_skill_learning.models.model_utils import *
from libero.lifelong.algos.lotus_skill_learning.models.torch_utils import to_onehot
class SubtaskSequenceDataset(Dataset):
    def __init__(self,
                 data_file_list,
                 subtask_file_list,
                 subtask_id,
                 data_modality=["image", "proprio"],             
                 use_eye_in_hand=True,
                 use_subgoal_eye_in_hand=False,
                 subgoal_cfg=None,
                 seq_len=10,
                 task_embs=None,
                 goal_modality="BUDS",
                 dinov2_file_list=[],
                 new_task_name="@default@",
                 demo_range=range(0, 50)):
        # demo_num = data_file["data"].attrs["demo_num"]
        self.dataset_num = len(data_file_list)
        self.data_modality = data_modality
        self.goal_modality = goal_modality
        self.use_eye_in_hand = use_eye_in_hand
        self.use_subgoal_eye_in_hand = use_subgoal_eye_in_hand
        self.subtask_id = subtask_id

        self.subgoal_cfg = subgoal_cfg
        
        self._idx_to_seg_id = dict()
        self._seg_id_to_start_indices = dict()
        self._seg_id_to_seg_length = dict()

        self.seq_length = seq_len

        self.agentview_image_names = []
        self.frontview_image_names = []
        self.eye_in_hand_image_names = []
        self.goal_image_names = []

        self.actions = []
        self.states = []

        self.agentview_images = []
        self.eye_in_hand_images = []
        self.gripper_states = []
        self.joint_states = []
        self.ee_states = []
        self.goal_images = []
        self.dinov2_features = []
        self.subgoal_indices = []


        self.proprios = []
        start_idx = 0 # Clip initial few steps of each episode
        self.total_len = 0
        count = 0
        self.not_use_this_dataset = False

        if not dinov2_file_list:
            for file_id, (data_file, subtask_file) in enumerate(zip(data_file_list, subtask_file_list)):
                subtask_segmentation = subtask_file["subtasks"][f"subtask_{subtask_id}"]["segmentation"][()]
                for (seg_idx, (i, start_idx, end_idx)) in enumerate(subtask_segmentation):
                    if isinstance(new_task_name, list):
                        if any(name in data_file.filename for name in new_task_name):
                            demo_range = range(0, new_task_demo_num)
                        else:
                            demo_range = range(0, old_task_demo_num)
                    else:
                        if new_task_name in data_file.filename:
                            demo_range = range(0, new_task_demo_num)# range(30, 50)
                        else:
                            demo_range = range(0, old_task_demo_num)# range(40, 50)

                    if i not in demo_range:
                        continue
                    agentview_images = data_file[f"data/demo_{i}/obs/agentview_rgb"][()][start_idx:end_idx+1]
                    eye_in_hand_images = data_file[f"data/demo_{i}/obs/eye_in_hand_rgb"][()][start_idx:end_idx+1]

                    self._seg_id_to_start_indices[(file_id, seg_idx)] = self.total_len
                    self._seg_id_to_seg_length[(file_id, seg_idx)] = end_idx - start_idx + 1

                    actions = data_file[f"data/demo_{i}/actions"][()][start_idx:end_idx+1]
                    gripper_states = data_file[f"data/demo_{i}/obs/gripper_states"][()][start_idx:end_idx+1]
                    joint_states = data_file[f"data/demo_{i}/obs/joint_states"][()][start_idx:end_idx+1]
                    ee_states = data_file[f"data/demo_{i}/obs/ee_states"][()][start_idx:end_idx+1]
                    
                    for j in range(end_idx - start_idx + 1):
                        self._idx_to_seg_id[self.total_len] = (file_id, seg_idx)
                        self.total_len += 1
                        self.agentview_images.append(torch.from_numpy(np.array(agentview_images[j]).transpose(2, 0, 1)))
                        self.eye_in_hand_images.append(torch.from_numpy(np.array(eye_in_hand_images[j]).transpose(2, 0, 1)))
                        future_idx = min(end_idx, start_idx + j + subgoal_cfg["horizon"]) - start_idx
                        self.subgoal_indices.append(future_idx + count)
                        
                    count = len(self.subgoal_indices)
                    self.actions.append(actions)
                    self.gripper_states.append(gripper_states)
                    self.joint_states.append(joint_states)
                    self.ee_states.append(ee_states)

            if len(self.actions) == 0:
                self.not_use_this_dataset = True
                return None
            self.actions = np.vstack(self.actions)
            self.actions = safe_cuda(torch.from_numpy(self.actions))
            self.gripper_states = np.vstack(self.gripper_states)
            self.gripper_states = safe_cuda(torch.from_numpy(self.gripper_states))
            self.joint_states = np.vstack(self.joint_states)
            self.joint_states = safe_cuda(torch.from_numpy(self.joint_states))
            self.ee_states = np.vstack(self.ee_states)
            self.ee_states = safe_cuda(torch.from_numpy(self.ee_states))
            self.agentview_images = safe_cuda(torch.stack(self.agentview_images, dim=0))
            self.eye_in_hand_images = safe_cuda(torch.stack(self.eye_in_hand_images, dim=0))
            assert(len(self.actions) == len(self.subgoal_indices))
            assert(max(self.subgoal_indices) == len(self.actions)-1)
        
        else: # use dinov2 features
            for file_id, (data_file, subtask_file, dinov2_file) in enumerate(zip(data_file_list, subtask_file_list, dinov2_file_list)):
                subtask_segmentation = subtask_file["subtasks"][f"subtask_{subtask_id}"]["segmentation"][()]
                for (seg_idx, (i, start_idx, end_idx)) in enumerate(subtask_segmentation):
                    if isinstance(new_task_name, list):
                        if any(name in data_file.filename for name in new_task_name):
                            demo_range = range(0, new_task_demo_num)
                        else:
                            demo_range = range(0, old_task_demo_num)
                    else:
                        if new_task_name in data_file.filename:
                            demo_range = range(0, new_task_demo_num)# range(30, 50)
                        else:
                            demo_range = range(0, old_task_demo_num)# range(40, 50)
                    
                    if i not in demo_range:
                        continue
                    agentview_images = data_file[f"data/demo_{i}/obs/agentview_rgb"][()][start_idx:end_idx+1]
                    eye_in_hand_images = data_file[f"data/demo_{i}/obs/eye_in_hand_rgb"][()][start_idx:end_idx+1]
                    dinov2_features = dinov2_file[f"data/demo_{i}/embedding"][()][start_idx:end_idx+1]

                    self._seg_id_to_start_indices[(file_id, seg_idx)] = self.total_len
                    self._seg_id_to_seg_length[(file_id, seg_idx)] = end_idx - start_idx + 1

                    actions = data_file[f"data/demo_{i}/actions"][()][start_idx:end_idx+1]
                    gripper_states = data_file[f"data/demo_{i}/obs/gripper_states"][()][start_idx:end_idx+1]
                    joint_states = data_file[f"data/demo_{i}/obs/joint_states"][()][start_idx:end_idx+1]
                    ee_states = data_file[f"data/demo_{i}/obs/ee_states"][()][start_idx:end_idx+1]
                    
                    for j in range(end_idx - start_idx + 1):
                        self._idx_to_seg_id[self.total_len] = (file_id, seg_idx)
                        self.total_len += 1
                        self.agentview_images.append(torch.from_numpy(np.array(agentview_images[j]).transpose(2, 0, 1)))
                        self.eye_in_hand_images.append(torch.from_numpy(np.array(eye_in_hand_images[j]).transpose(2, 0, 1)))
                        future_idx = min(end_idx, start_idx + j + subgoal_cfg["horizon"]) - start_idx
                        self.subgoal_indices.append(future_idx + count)
                        
                    count = len(self.subgoal_indices)
                    self.actions.append(actions)
                    self.gripper_states.append(gripper_states)
                    self.joint_states.append(joint_states)
                    self.ee_states.append(ee_states)
                    self.dinov2_features.append(dinov2_features)

            if len(self.actions) == 0:
                self.not_use_this_dataset = True
                return None
            self.actions = np.vstack(self.actions)
            self.actions = safe_cuda(torch.from_numpy(self.actions))
            self.gripper_states = np.vstack(self.gripper_states)
            self.gripper_states = safe_cuda(torch.from_numpy(self.gripper_states))
            self.joint_states = np.vstack(self.joint_states)
            self.joint_states = safe_cuda(torch.from_numpy(self.joint_states))
            self.ee_states = np.vstack(self.ee_states)
            self.ee_states = safe_cuda(torch.from_numpy(self.ee_states))
            self.dinov2_features = np.vstack(self.dinov2_features)
            self.dinov2_features = safe_cuda(torch.from_numpy(self.dinov2_features))
            self.agentview_images = safe_cuda(torch.stack(self.agentview_images, dim=0))
            self.eye_in_hand_images = safe_cuda(torch.stack(self.eye_in_hand_images, dim=0))
            assert(len(self.actions) == len(self.subgoal_indices))
            assert(len(self.dinov2_features) == len(self.actions))
            assert(max(self.subgoal_indices) == len(self.actions)-1)
            
        print(f"Finish loading subtask_{subtask_id}: ", self.total_len)

    @property
    def action_dim(self):
        return self.actions.shape[-1]


    @property
    def proprio_dim(self):
        if self.proprios == []:
            return 0
        else:
            return self.proprios.shape[-1]
    
    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        data={}
        data["obs"]={}
        file_id, seg_id = self._idx_to_seg_id[idx]
        seg_start_index = self._seg_id_to_start_indices[(file_id, seg_id)]
        seg_length = self._seg_id_to_seg_length[(file_id, seg_id)]

        index_in_seg = idx - seg_start_index
        end_index_in_seg = seg_length

        seq_begin_index = max(0, index_in_seg)
        seq_end_index = min(seg_length, index_in_seg + self.seq_length)
        padding = max(0, seq_begin_index + self.seq_length - seg_length)

        seq_begin_index += seg_start_index
        seq_end_index += seg_start_index
        
        action_seq = self.actions[seq_begin_index: seq_end_index].float()
        gripper_state_seq = self.gripper_states[seq_begin_index: seq_end_index].float()
        joint_state_seq = self.joint_states[seq_begin_index: seq_end_index].float()
        ee_state_seq = self.ee_states[seq_begin_index: seq_end_index].float()

        if self.goal_modality != "dinov2":
            agentview_seq = self.agentview_images[seq_begin_index: seq_end_index]
            eye_in_hand_seq = self.eye_in_hand_images[seq_begin_index: seq_end_index]
            # use single subgoal or seq subgoals in the sequence
            subgoal_index = self.subgoal_indices[seq_end_index-1] #TODO: need to reconsider this
            subgoal_seq = self.subgoal_indices[seq_begin_index: seq_end_index]
            if padding > 0:
                # Pad
                action_end_pad = torch.repeat_interleave(action_seq[-1].unsqueeze(0), padding, dim=0)
                action_seq = torch.cat([action_seq] + [action_end_pad], dim=0)

                gripper_state_end_pad = torch.repeat_interleave(gripper_state_seq[-1].unsqueeze(0), padding, dim=0)
                gripper_state_seq = torch.cat([gripper_state_seq] + [gripper_state_end_pad], dim=0)

                joint_state_end_pad = torch.repeat_interleave(joint_state_seq[-1].unsqueeze(0), padding, dim=0)
                joint_state_seq = torch.cat([joint_state_seq] + [joint_state_end_pad], dim=0)

                ee_state_end_pad = torch.repeat_interleave(ee_state_seq[-1].unsqueeze(0), padding, dim=0)
                ee_state_seq = torch.cat([ee_state_seq] + [ee_state_end_pad], dim=0)

                agentview_end_pad = torch.repeat_interleave(agentview_seq[-1].unsqueeze(0), padding, dim=0)
                agentview_seq = torch.cat([agentview_seq] + [agentview_end_pad], dim=0)

                eye_in_hand_end_pad = torch.repeat_interleave(eye_in_hand_seq[-1].unsqueeze(0), padding, dim=0)
                eye_in_hand_seq = torch.cat([eye_in_hand_seq] + [eye_in_hand_end_pad], dim=0)

                subgoal_end_pad = [subgoal_seq[-1]] * padding
                subgoal_seq = subgoal_seq + subgoal_end_pad


            if self.use_eye_in_hand:
                agentview_rgb = agentview_seq.float() / 255.
                eye_in_hand_rgb = eye_in_hand_seq.float() / 255.
                data["obs"]["agentview_rgb"] = agentview_rgb
                data["obs"]["eye_in_hand_rgb"] = eye_in_hand_rgb    
            else:
                agentview_rgb = agentview_seq.float() / 255.
                data["obs"]["agentview_rgb"] = agentview_rgb

            # if self.goal_modality == "BUDS":
            if self.use_subgoal_eye_in_hand:
                # TODO:need to update
                subgoal = torch.cat((self.agentview_images[subgoal_index],
                                    self.eye_in_hand_images[subgoal_index]), dim=1).float() / 255.
                data["obs"]["subgoal"] = subgoal
            else:
                # # use individual subgoal in the sequence
                # subgoal = [self.agentview_images[i] for i in subgoal_seq]
                # data["obs"]["subgoal"] = torch.stack(subgoal, dim=0).float() / 255.

                # repeat final subgoal in the sequence
                subgoal = self.agentview_images[subgoal_index].float() / 255.
                data["obs"]["subgoal"] = subgoal.unsqueeze(0).repeat(self.seq_length, 1, 1, 1)

            # elif self.goal_modality == "ee_states":
            #     # # use individual subgoal in the sequence
            #     # subgoal = [torch.cat([self.ee_states[i]] + [self.gripper_states[i]], dim=0) for i in subgoal_seq]
            #     # data["obs"]["subgoal"] = torch.stack(subgoal, dim=0).float()

            #     # repeat final subgoal in the sequence
            #     subgoal = torch.cat([self.ee_states[subgoal_index]] + [self.gripper_states[subgoal_index]], dim=0)
            #     data["obs"]["subgoal"] = subgoal.unsqueeze(0).repeat(self.seq_length, 1)

            # elif self.goal_modality == "joint_states":
            #     # # use individual subgoal in the sequence
            #     # subgoal = [torch.cat([self.joint_states[i]] + [self.gripper_states[i]], dim=0) for i in subgoal_seq]
            #     # data["obs"]["subgoal"] = torch.stack(subgoal, dim=0).float()

            #     # repeat final subgoal in the sequence
            #     subgoal = torch.cat([self.joint_states[subgoal_index]] + [self.gripper_states[subgoal_index]], dim=0)
            #     data["obs"]["subgoal"] = subgoal.unsqueeze(0).repeat(self.seq_length, 1)

            data["actions"] = action_seq
            data['obs']["gripper_states"] = gripper_state_seq
            data['obs']["joint_states"] = joint_state_seq
            return data

        else: # use dinov2 features
            dinov2_feature_seq = self.dinov2_features[seq_begin_index: seq_end_index].float()
            agentview_seq = self.agentview_images[seq_begin_index: seq_end_index]
            eye_in_hand_seq = self.eye_in_hand_images[seq_begin_index: seq_end_index]
            # use single subgoal or seq subgoals in the sequence
            subgoal_index = self.subgoal_indices[seq_end_index-1] #TODO: need to reconsider this
            subgoal_seq = self.subgoal_indices[seq_begin_index: seq_end_index]
            if padding > 0:
                # Pad
                action_end_pad = torch.repeat_interleave(action_seq[-1].unsqueeze(0), padding, dim=0)
                action_seq = torch.cat([action_seq] + [action_end_pad], dim=0)

                gripper_state_end_pad = torch.repeat_interleave(gripper_state_seq[-1].unsqueeze(0), padding, dim=0)
                gripper_state_seq = torch.cat([gripper_state_seq] + [gripper_state_end_pad], dim=0)

                joint_state_end_pad = torch.repeat_interleave(joint_state_seq[-1].unsqueeze(0), padding, dim=0)
                joint_state_seq = torch.cat([joint_state_seq] + [joint_state_end_pad], dim=0)

                ee_state_end_pad = torch.repeat_interleave(ee_state_seq[-1].unsqueeze(0), padding, dim=0)
                ee_state_seq = torch.cat([ee_state_seq] + [ee_state_end_pad], dim=0)

                agentview_end_pad = torch.repeat_interleave(agentview_seq[-1].unsqueeze(0), padding, dim=0)
                agentview_seq = torch.cat([agentview_seq] + [agentview_end_pad], dim=0)

                eye_in_hand_end_pad = torch.repeat_interleave(eye_in_hand_seq[-1].unsqueeze(0), padding, dim=0)
                eye_in_hand_seq = torch.cat([eye_in_hand_seq] + [eye_in_hand_end_pad], dim=0)

                dinov2_feature_end_pad = torch.repeat_interleave(dinov2_feature_seq[-1].unsqueeze(0), padding, dim=0)
                dinov2_feature_seq = torch.cat([dinov2_feature_seq] + [dinov2_feature_end_pad], dim=0)

                subgoal_end_pad = [subgoal_seq[-1]] * padding
                subgoal_seq = subgoal_seq + subgoal_end_pad


            if self.use_eye_in_hand:
                agentview_rgb = agentview_seq.float() / 255.
                eye_in_hand_rgb = eye_in_hand_seq.float() / 255.
                data["obs"]["agentview_rgb"] = agentview_rgb
                data["obs"]["eye_in_hand_rgb"] = eye_in_hand_rgb    
            else:
                agentview_rgb = agentview_seq.float() / 255.
                data["obs"]["agentview_rgb"] = agentview_rgb

            if self.goal_modality == "dinov2":
                # # use individual subgoal in the sequence
                # subgoal = [self.dinov2_features[i] for i in subgoal_seq]
                # data["obs"]["subgoal"] = torch.stack(subgoal, dim=0).float()

                # repeat final subgoal in the sequence
                subgoal = self.dinov2_features[subgoal_index].float()
                data["obs"]["subgoal"] = subgoal.repeat(self.seq_length, 1)
            else:
                pass ## TODO

            data["actions"] = action_seq
            data['obs']["gripper_states"] = gripper_state_seq
            data['obs']["joint_states"] = joint_state_seq
            return data


class SkillLearningDataset():
    def __init__(self,
                 data_file_name_list,
                 subtasks_file_name_list,
                 data_modality=["image", "proprio"],
                 use_eye_in_hand=True,
                 subgoal_cfg=None,
                 subtask_id=[],
                 seq_len=10,
                 task_embs=None,
                 used_data_file_name_list=[],
                 goal_modality="BUDS",
                 new_task_name="@default@",
                 demo_range=range(0, 50),
                 ):
    
        self.f_list = []
        self.dinov2_f_list = []
        self.train_dataset_id = []
        self.new_task_name = new_task_name
        self.demo_range = demo_range
        self.goal_modality = goal_modality
        for data_file_name in data_file_name_list:
            if any(used_data_file_name in data_file_name for used_data_file_name in used_data_file_name_list):
                self.f_list.append(h5py.File(data_file_name, "r"))
                if goal_modality == "dinov2":
                    import re
                    dinov2_feature_file_name = re.sub(r"(datasets/)([^/]+)(/)", r"\1dinov2/\2\3", data_file_name)
                    self.dinov2_f_list.append(h5py.File(dinov2_feature_file_name, "r"))

        self.subtasks_f_list = []
        for subtasks_file_name in subtasks_file_name_list:
            if any(used_data_file_name in subtasks_file_name for used_data_file_name in used_data_file_name_list):
                self.subtasks_f_list.append(h5py.File(subtasks_file_name, "r"))

        self.subtask_id = subtask_id
        self.data_modality = data_modality
        self.use_eye_in_hand = use_eye_in_hand
        self.num_subtasks = self.subtasks_f_list[0]["subtasks"].attrs["num_subtasks"]
        self.subgoal_cfg = subgoal_cfg
        self.seq_len = seq_len
        self.task_embs = task_embs

        for subtasks_f in self.subtasks_f_list:
            print("subtasks distance score:",subtasks_f["subtasks"].attrs["score"])
            if isinstance(new_task_name, list):
                for task in new_task_name:
                    if task == "@default@":
                        self.train_dataset_id = list(range(self.num_subtasks))
                    if task in subtasks_f.filename:
                        for key in subtasks_f['subtasks']:
                            if 'segmentation' in subtasks_f['subtasks'][key]:
                                data = subtasks_f['subtasks'][key]['segmentation'][()]
                                if data.size != 0:
                                    x = key.split('_')[-1]
                                    self.train_dataset_id.append(int(x))  
            else:
                if new_task_name == "@default@":
                    self.train_dataset_id = list(range(self.num_subtasks))
                if new_task_name in subtasks_f.filename:
                    for key in subtasks_f['subtasks']:
                        if 'segmentation' in subtasks_f['subtasks'][key]:
                            data = subtasks_f['subtasks'][key]['segmentation'][()]
                            if data.size != 0:
                                x = key.split('_')[-1]
                                self.train_dataset_id.append(int(x))
            self.train_dataset_id = sorted(list(set(self.train_dataset_id)))
        print("train_dataset_id:", self.train_dataset_id)
        self.datasets = []

    def get_dataset(self, idx):
        if self.subtask_id != []:
            if idx not in self.subtask_id:
                return None

        dataset = SubtaskSequenceDataset(self.f_list,
                                        self.subtasks_f_list,
                                        idx,
                                        data_modality=self.data_modality,
                                        use_eye_in_hand=self.use_eye_in_hand,
                                        use_subgoal_eye_in_hand=self.subgoal_cfg.use_eye_in_hand,
                                        subgoal_cfg=self.subgoal_cfg,
                                        seq_len=self.seq_len,
                                        task_embs=self.task_embs,
                                        goal_modality=self.goal_modality,
                                        dinov2_file_list=self.dinov2_f_list,
                                        new_task_name=self.new_task_name,
                                        demo_range=self.demo_range,)
        if dataset.not_use_this_dataset:
            return None
        # print(idx, len(dataset))
        return dataset
    
    def close(self):
        for f in self.f_list:
            f.close()
        for subtasks_f in self.subtasks_f_list:
            self.subtasks_f.close()


class MetaPolicyDataset(Dataset):
    def __init__(self,
                 data_file_name_list,
                 embedding_file_name,
                 subtasks_file_name_list,
                 task_names,
                 task_embs,
                 use_eye_in_hand=False,
                 ):

        embedding_file = h5py.File(embedding_file_name, "r")
        self.use_eye_in_hand = use_eye_in_hand
        self.num_subtasks = h5py.File(subtasks_file_name_list[0], "r")["subtasks"].attrs["num_subtasks"]
        self.demo_num = h5py.File(subtasks_file_name_list[0], "r")["subtasks"].attrs["demo_num"]

        self.embeddings = []
        self.goal_embeddings = []

        self.agentview_image_names = []
        self.eye_in_hand_image_names = []
        self.subgoal_embeddings = []

        self.subtask_labels = []
        self.task_idx = []
        self.task_embs = task_embs

        self.agentview_images = []
        self.eye_in_hand_images = []

        self.total_len = 0

        for data_file_name, subtasks_file_name in zip(data_file_name_list, subtasks_file_name_list):
            data_file = h5py.File(data_file_name, "r")
            dataset_category, dataset_name = data_file_name.split("/")[-2:]
            dataset_name = dataset_name.split(".")[0]
            subtasks_file = h5py.File(subtasks_file_name, "r")
            task_idx = task_names.index(dataset_name.split("_demo")[0])

            for ep_idx in range(self.demo_num):
                try:
                    saved_ep_subtasks_seq = subtasks_file["subtasks"][f"demo_subtasks_seq_{ep_idx}"][()]
                except:
                    continue
                for (k, (start_idx, end_idx, subtask_label)) in enumerate(saved_ep_subtasks_seq):
                    if k == len(saved_ep_subtasks_seq) - 1:
                        e_idx = end_idx + 1
                    else:
                        e_idx = end_idx
                    agentview_images = data_file[f"data/demo_{ep_idx}/obs/agentview_rgb"][()][start_idx:e_idx]
                    eye_in_hand_images = data_file[f"data/demo_{ep_idx}/obs/eye_in_hand_rgb"][()][start_idx:e_idx]

                    embeddings = embedding_file[f"{dataset_name}/demo_{ep_idx}/embedding"][()][start_idx:e_idx]
                    for j in range(len(agentview_images)):
                        self.agentview_images.append(torch.from_numpy(np.array(agentview_images[j]).transpose(2, 0, 1)))
                        self.eye_in_hand_images.append(torch.from_numpy(np.array(eye_in_hand_images[j]).transpose(2, 0, 1)))
                        self.subgoal_embeddings.append(torch.from_numpy(embeddings[j]))
                        
                        self.subtask_labels.append(subtask_label)
                        self.task_idx.append(task_idx)
                        self.total_len += 1
            
            data_file.close()
        embedding_file.close()


        self.subgoal_embedding_dim =  len(self.subgoal_embeddings[-1])
         
        self.agentview_images = safe_cuda(torch.stack(self.agentview_images, dim=0))
        self.eye_in_hand_images = safe_cuda(torch.stack(self.eye_in_hand_images, dim=0))
        self.subgoal_embeddings = safe_cuda(torch.stack(self.subgoal_embeddings, dim=0))

        assert(self.total_len == len(self.subtask_labels))
        self.subtask_labels = safe_cuda(torch.from_numpy(np.array(self.subtask_labels)))
        
        # print(self.agentview_images.shape)
        print("MetaPolicyDataset: ", self.subtask_labels.shape)
        embedding_file.close()

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        agentview_image = self.agentview_images[idx].float() / 255.
        if self.use_eye_in_hand:
            eye_in_hand_image = self.eye_in_hand_images[idx].float() / 255.
        #     state_image = torch.cat([agentview_image, eye_in_hand_image], dim=0)
        # else:
        #     state_image = agentview_image
        subgoal_embedding = self.subgoal_embeddings[idx].float()
        subtask_label = self.subtask_labels[idx]
        task_idx = self.task_idx[idx]
        task_emb = self.task_embs[task_idx]
        # return {"state_image": state_image, "embedding": subgoal_embedding, "id_vector": to_onehot(subtask_label, self.num_subtasks)},{"embedding": subgoal_embedding, "id": subtask_label}
        data = {}
        data["obs"] = {"agentview_rgb": agentview_image, "task_emb": task_emb, "embedding": subgoal_embedding, "id_vector": to_onehot(subtask_label, self.num_subtasks), "id": subtask_label}
        return data


class MetaPolicySequenceDataset(Dataset):
    def __init__(self,
                 data_file_name_list,
                 embedding_file_name,
                 subtasks_file_name_list,
                 task_names,
                 task_embs,
                 use_eye_in_hand=False,
                 seq_length=10,
                 new_task_name="@default@",
                 demo_range=range(0, 50),
                 used_data_file_name_list=[],
                 ):

        embedding_file = h5py.File(embedding_file_name, "r")
        self.use_eye_in_hand = use_eye_in_hand
        self.seq_length = seq_length
        self.num_subtasks = h5py.File(subtasks_file_name_list[0], "r")["subtasks"].attrs["num_subtasks"]
        self.demo_num = h5py.File(subtasks_file_name_list[0], "r")["subtasks"].attrs["demo_num"]

        self.embeddings = []
        self.goal_embeddings = []

        self.agentview_image_names = []
        self.eye_in_hand_image_names = []
        self.subgoal_embeddings = []

        self.subtask_labels = []
        self.task_idx = []
        self.task_embs = task_embs

        self.agentview_images = []
        self.eye_in_hand_images = []

        self.total_len = 0
        self._idx_to_seg_id = dict()
        self._seg_id_to_start_indices = dict()
        self._seg_id_to_seg_length = dict()
        seg_idx = 0

        # demo_num_10_file_names =[
        #     "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        #     "pick_up_the_cream_cheese_and_place_it_in_the_basket",
        #     "pick_up_the_salad_dressing_and_place_it_in_the_basket",

        #     "put_the_bowl_on_the_stove",
        #     "put_the_cream_cheese_in_the_bowl",
        # ]

        for data_file_name, subtasks_file_name in zip(data_file_name_list, subtasks_file_name_list):
            if not any(used_data_file_name + "_demo" in data_file_name for used_data_file_name in used_data_file_name_list):
                continue
            dataset_category, dataset_name = data_file_name.split("/")[-2:]
            dataset_name = dataset_name.split(".")[0]
            try:
                task_idx = task_names.index(dataset_name.split("_demo")[0])
            except:
                print("task name not found")
                continue
            data_file = h5py.File(data_file_name, "r")
            subtasks_file = h5py.File(subtasks_file_name, "r")
            # print(used_data_file_name_list)
            # print(data_file_name + ", " + subtasks_file_name)

            if isinstance(new_task_name, list):
                if any(name in data_file.filename for name in new_task_name):
                    demo_range = range(0, new_task_demo_num)
                else:
                    demo_range = range(0, old_task_demo_num)
            else:
                if new_task_name in data_file.filename:
                    demo_range = range(0, new_task_demo_num)# range(30, 50)
                else:
                    demo_range = range(0, old_task_demo_num)# range(40, 50)

            for ep_idx in range(self.demo_num):
                if ep_idx not in demo_range:
                    continue
                try:
                    saved_ep_subtasks_seq = subtasks_file["subtasks"][f"demo_subtasks_seq_{ep_idx}"][()]
                except:
                    continue
                for (k, (start_idx, end_idx, subtask_label)) in enumerate(saved_ep_subtasks_seq):
                    if k == len(saved_ep_subtasks_seq) - 1:
                        e_idx = end_idx + 1
                    else:
                        e_idx = end_idx
                    self._seg_id_to_start_indices[seg_idx] = self.total_len
                    self._seg_id_to_seg_length[seg_idx] = end_idx - start_idx + 1
                    agentview_images = data_file[f"data/demo_{ep_idx}/obs/agentview_rgb"][()][start_idx:e_idx]
                    eye_in_hand_images = data_file[f"data/demo_{ep_idx}/obs/eye_in_hand_rgb"][()][start_idx:e_idx]

                    embeddings = embedding_file[f"{dataset_name}/demo_{ep_idx}/embedding"][()][start_idx:e_idx]
                    for j in range(len(agentview_images)):
                        self.agentview_images.append(torch.from_numpy(np.array(agentview_images[j]).transpose(2, 0, 1)))
                        self.eye_in_hand_images.append(torch.from_numpy(np.array(eye_in_hand_images[j]).transpose(2, 0, 1)))
                        self.subgoal_embeddings.append(torch.from_numpy(embeddings[j]))
                        
                        self.subtask_labels.append(subtask_label)
                        self.task_idx.append(task_idx)
                        self._idx_to_seg_id[self.total_len] = seg_idx
                        self.total_len += 1
                    seg_idx += 1
            
            data_file.close()
        embedding_file.close()


        self.subgoal_embedding_dim =  len(self.subgoal_embeddings[-1])
         
        self.agentview_images = safe_cuda(torch.stack(self.agentview_images, dim=0))
        self.eye_in_hand_images = safe_cuda(torch.stack(self.eye_in_hand_images, dim=0))
        self.subgoal_embeddings = safe_cuda(torch.stack(self.subgoal_embeddings, dim=0))

        assert(self.total_len == len(self.subtask_labels))
        self.subtask_labels = safe_cuda(torch.from_numpy(np.array(self.subtask_labels)))
        
        # print(self.agentview_images.shape)
        print("MetaPolicyDataset: ", self.subtask_labels.shape)
        embedding_file.close()

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        seg_id = self._idx_to_seg_id[idx]
        seg_start_index = self._seg_id_to_start_indices[seg_id]
        seg_length = self._seg_id_to_seg_length[seg_id]

        index_in_seg = idx - seg_start_index
        end_index_in_seg = seg_length

        seq_begin_index = max(0, index_in_seg)
        seq_end_index = min(seg_length, index_in_seg + self.seq_length)
        padding = max(0, seq_begin_index + self.seq_length - seg_length)

        seq_begin_index += seg_start_index
        seq_end_index += seg_start_index

        agentview_seq = self.agentview_images[seq_begin_index:seq_end_index]
        eye_in_hand_seq = self.eye_in_hand_images[seq_begin_index:seq_end_index]
        subgoal_embedding_seq = self.subgoal_embeddings[seq_begin_index:seq_end_index]
        subtask_label_seq = self.subtask_labels[seq_begin_index:seq_end_index]
        task_idx_seq = self.task_idx[seq_begin_index:seq_end_index]

        if padding > 0:
            agentview_end_pad = torch.repeat_interleave(agentview_seq[-1].unsqueeze(0), padding, dim=0)
            agentview_seq = torch.cat([agentview_seq] + [agentview_end_pad], dim=0)

            eye_in_hand_end_pad = torch.repeat_interleave(eye_in_hand_seq[-1].unsqueeze(0), padding, dim=0)
            eye_in_hand_seq = torch.cat([eye_in_hand_seq] + [eye_in_hand_end_pad], dim=0)

            subgoal_embedding_end_pad = torch.repeat_interleave(subgoal_embedding_seq[-1].unsqueeze(0), padding, dim=0)
            subgoal_embedding_seq = torch.cat([subgoal_embedding_seq] + [subgoal_embedding_end_pad], dim=0)

            subtask_label_end_pad = torch.repeat_interleave(subtask_label_seq[-1].unsqueeze(0), padding, dim=0)
            subtask_label_seq = torch.cat([subtask_label_seq] + [subtask_label_end_pad], dim=0)

            task_idx_end_pad = [task_idx_seq[-1]] * padding
            task_idx_seq.extend(task_idx_end_pad)

        
        agentview_seq = agentview_seq.float() / 255.
        subgoal_embedding_seq = subgoal_embedding_seq.float()
        task_emb_seq = [self.task_embs[task_idx] for task_idx in task_idx_seq]

        data = {}
        data["task_emb"] = task_emb_seq[0]
        data["obs"] = {"agentview_rgb": agentview_seq, "embedding": subgoal_embedding_seq, "id_vector": to_onehot(subtask_label_seq, self.num_subtasks), "id": subtask_label_seq}
        return data