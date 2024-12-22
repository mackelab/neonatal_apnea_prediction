import logging
import os
from datetime import datetime
from random import shuffle

import numpy as np
import pandas as pd
import scipy
import torch

from pyedflib import highlevel
from scipy.signal import decimate
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler

log = logging.getLogger(__name__)


# Dict for reading in the signals, together with the sampling frequency
SIGNAL_TYPE = {
    "EKG": ("EKG", 200),
    "Pleth Radical": ("PR", 100),
    "Nasal Pressure": ("NP", 200),
    "Thorax": ("Thorax", 50),
    "Abdomen": ("Abdomen", 50),
    "Sentec PCO2": ("PCO2", 2),
    "SpO2 Radical": ("SpO2", 2),
    "Heart Rate B2B": ("HR", 1),
    "Heart rate B2B": ("HR", 1),  # One file contains "rate" instead of "Rate"
}


# Read in list of raw signals. Create pickle files for faster access.
def read_data(pat_id, file_path):
    print(f"Reading ID {pat_id}.")

    signals, signal_headers, header = highlevel.read_edf(
        os.path.join(file_path, f"signals{pat_id}.edf")
    )

    all_signals = {}

    for sig_header, sig in zip(signal_headers, signals):
        if sig_header["label"] in SIGNAL_TYPE.keys():
            key, freq = SIGNAL_TYPE[sig_header["label"]]
            all_signals[key] = np.repeat(sig, 200 // freq)
        else:
            # Log skipped signal types.
            print(f"Skipped { sig_header['label'] }")
            continue

    annotations = pd.read_csv(
        os.path.join(file_path, f"annotations{pat_id}.txt"),
        sep="\t",
    )

    annotation_types = [
        "APNEA-OBSTRUCTIVE",
        "APNEA-CENTRAL",
        "APNEA-MIXED",
        "HYPOPNEA-OBSTRUCTIVE",
        "HYPOPNEA-CENTRAL",
        "HYPOPNEA-MIXED",
        "DESAT",
        "ACTIVITY-MOVE",
        "SIGNAL-ARTIFACT",
        "SIGNAL-QUALITY-LOW",
    ]
    all_signals.update(
        {
            anno_type: np.zeros_like(signals[0], dtype=int)
            for anno_type in annotation_types
        }
    )

    SIGNAL_START = header["startdate"].timestamp()
    annotation_start = None
    annotation_stop = None
    for row in annotations.itertuples(index=False):
        event_type = row.type
        start = datetime.strptime(row.timestamp, "%Y-%m-%dT%H:%M:%S.%f").timestamp()
        dur = row.duration
        # Frequency 200Hz: 1 / 200 = 0.005
        ind_start = int(np.round((start - SIGNAL_START) / 0.005))
        ind_end = int(np.round((start + dur - SIGNAL_START) / 0.005))
        if event_type == "ANALYSIS-START":
            assert ind_start == ind_end
            annotation_start = ind_start
        elif event_type == "ANALYSIS-STOP":
            assert ind_start == ind_end
            annotation_stop = ind_end
        elif event_type in annotation_types:
            all_signals[event_type][ind_start : ind_end + 1] = 1
        else:
            raise ValueError

    return {k: v[annotation_start:annotation_stop] for k, v in all_signals.items()}


def sum_from_left(arr):
    res = np.zeros_like(arr)
    sumi = 0
    for idx, i in enumerate(arr):
        if i == 0:
            sumi = 0
        if i == 1:
            res[idx] = sumi
            sumi += 1
    return res


def sum_from_right(arr):
    temp = sum_from_left(arr[::-1])
    return temp[::-1]


def create_slices(cutter):
    temp = np.concatenate([np.zeros(1), cutter, np.zeros(1)])

    diff = temp[1:] - temp[:-1]
    start_pos_arr = np.where(diff == 1)[0]
    end_pos_arr = np.where(diff == -1)[0]

    return [slice(start, end) for start, end in zip(start_pos_arr, end_pos_arr)]


def standardize(arr):
    arr_mean = np.mean(arr)
    arr_std = np.std(arr)
    return (arr - arr_mean) / (arr_std if (arr_std > 0.0) else 1.0)


def normalize_range(signal, lower_source, upper_source, lower_target, upper_target):
    assert lower_source < upper_source
    assert lower_target < upper_target

    return (
        (upper_target - lower_target)
        * ((signal - lower_source) / (upper_source - lower_source))
    ) + lower_target


def create_time_windows(anti_adverse_events, cutter, window_size, away, lag):
    assert away >= window_size + lag

    increasing = sum_from_left(anti_adverse_events)
    decreasing = sum_from_right(anti_adverse_events)
    slice_ls = create_slices(cutter)

    slices_and_labels = []
    for ss in slice_ls[:-1]:  # Throw away in last slice.
        decr_min = np.min(decreasing[ss])
        decr_max = np.max(decreasing[ss]) + 1
        incr_min = np.min(increasing[ss])
        incr_max = np.max(increasing[ss]) + 1
        if (decr_min <= lag) and (decr_max >= lag + window_size):
            start_pos = lag - decr_min
            slices_and_labels.append(
                {
                    "label": 1,
                    "slice": slice(
                        ss.stop - start_pos - window_size, ss.stop - start_pos
                    ),
                }
            )
        if (decr_max >= away + window_size) and (incr_max >= away + window_size):
            start_pos = ss.start + max(away - incr_min, 0)
            end_pos = ss.stop - max(away - decr_min, 0)
            slices_and_labels.extend(
                [
                    {
                        "label": 0,
                        "slice": slice(
                            end_pos - (window_size * (i + 1)),
                            end_pos - (window_size * i),
                        ),
                    }
                    for i in range((end_pos - start_pos) // window_size)
                ]
            )
    return slices_and_labels


class NeoNatal(Dataset):
    def __init__(
        self,
        pat_id,
        signal_dict,
        dataset_mode,
        signal_types,
        adverse_events,
        cutter_events,
        time_window,
        lag,
        away,
    ):
        super().__init__()
        self.pat_id = pat_id  # for logging
        self.dataset_mode = dataset_mode
        self.signal_types = signal_types
        self.adverse_events = adverse_events
        self.cutter_events = cutter_events
        self.time_window = time_window
        self.lag = lag
        self.away = away
        self.signal_dict = signal_dict

        # Adverse events on which distances are calulated on.
        self.adverse_events_aggregated = 1 - np.column_stack(
            [self.signal_dict[event] for event in self.adverse_events]
        ).max(axis=1)

        # Adverse events plus the events that should not be part of the set of time windows.
        cutter = 1 - np.column_stack(
            [
                self.signal_dict[event]
                for event in (self.adverse_events + self.cutter_events)
            ]
        ).max(axis=1)

        # Create the time window slices. Together with the labels.
        # No signals processed for now.
        self.time_window_df = pd.DataFrame(
            create_time_windows(
                anti_adverse_events=self.adverse_events_aggregated,
                cutter=cutter,
                window_size=self.time_window,
                away=self.away,
                lag=self.lag,
            )
        )

        log.info(f"Number of time windows: {len(self.time_window_df)}")
        log.info(
            f"Imbalance: {self.time_window_df['label'].sum() / len(self.time_window_df)}"
        )

        # Add the processed signal time windows here.
        if dataset_mode == "list" or dataset_mode == "stacked":
            self.time_window_df["sig"] = self.time_window_df.apply(
                lambda row: self.feature_extraction(row["slice"]), axis=1
            )
        elif dataset_mode == "features":
            self.time_window_df["sig"] = self.time_window_df.apply(
                lambda row: self.feature_engineering(row["slice"]), axis=1
            )
        else:
            raise ValueError

    # Get undersampled elements for training.
    def __getitem__(self, index):
        if self.dataset_mode == "stacked":
            return self._get_stacked(index)
        elif self.dataset_mode == "list":
            return self._get_list(index)
        elif self.dataset_mode == "features":
            return self._get_features(index)
        else:
            raise ValueError

    def __len__(self):
        return len(self.time_window_df)

    # Needs to be implemented for BalancedSamlping
    def _get_labels(self):
        return self.time_window_df["label"].to_numpy()

    # Stacked mode requires that all signals have the same length.
    # Or that only one channel is used.
    def _get_stacked(self, index):
        index_row = self.time_window_df.iloc[index]
        sig_list = index_row["sig"]
        if len(sig_list) > 1:
            max_len = max([len(sig) for sig in sig_list])
            sig_list = [np.repeat(sig, max_len // len(sig)) for sig in sig_list]
        x_s = torch.from_numpy(np.row_stack(sig_list).astype(np.float32))
        y_s = torch.Tensor([np.float32(index_row["label"])])
        return (x_s, y_s)

    def _get_list(self, index):
        index_row = self.time_window_df.iloc[index]
        x_s = [
            torch.from_numpy(ind.astype(np.float32)).unsqueeze(0)
            for ind in index_row["sig"]
        ]
        y_s = torch.Tensor([np.float32(index_row["label"])])
        return (x_s, y_s)

    def _get_features(self, index):
        index_row = self.time_window_df.iloc[index]
        x_s = torch.from_numpy(index_row["sig"].astype(np.float32))
        y_s = torch.Tensor([np.float32(index_row["label"])])
        return (x_s, y_s)

    def feature_extraction(self, ss):
        lower_range = -1.0
        upper_range = 1.0

        processed_windows = []
        for signal_type in self.signal_types:
            processed_window = None

            # 5 Hz
            if signal_type == "NP":
                np_decimate_one = 10
                np_decimate_two = 4
                processed_window = standardize(
                    decimate(
                        decimate(self.signal_dict["NP"][ss], np_decimate_one),
                        np_decimate_two,
                    )
                )
            # 5Hz
            elif signal_type == "Thorax":
                thorax_select = 4
                thorax_decimate = 10
                processed_window = standardize(
                    standardize(
                        decimate(
                            self.signal_dict["Thorax"][ss][::thorax_select],
                            thorax_decimate,
                        )
                    )
                    + standardize(
                        decimate(
                            self.signal_dict["Abdomen"][ss][::thorax_select],
                            thorax_decimate,
                        )
                    )
                )
            # 1Hz
            elif signal_type == "HR":
                hr_select = 200
                hr_lower = 50.0
                hr_upper = 240.0
                processed_window = normalize_range(
                    self.signal_dict["HR"][ss][::hr_select],  # normal mode
                    hr_lower,
                    hr_upper,
                    lower_range,
                    upper_range,
                )
            # 5Hz
            elif signal_type == "PR":
                pr_decimate_one = 10
                pr_decimate_two = 4
                processed_window = standardize(
                    decimate(
                        decimate(self.signal_dict["PR"][ss], pr_decimate_one),
                        pr_decimate_two,
                    )
                )
            # 1Hz
            elif signal_type == "SpO2":
                spo2_select = 100
                spo2_decimate = 2
                spo2_lower = 60.0
                spo2_upper = 100.0
                processed_window = normalize_range(
                    decimate(
                        self.signal_dict["SpO2"][ss][::spo2_select],
                        spo2_decimate,
                    ),
                    spo2_lower,
                    spo2_upper,
                    lower_range,
                    upper_range,
                )
            # 1Hz
            elif signal_type == "PCO2":
                pco2_select = 100
                pco2_decimate = 2
                pco2_lower = 30.0
                pco2_upper = 70.0
                processed_window = normalize_range(
                    decimate(
                        self.signal_dict["PCO2"][ss][::pco2_select],
                        pco2_decimate,
                    ),
                    pco2_lower,
                    pco2_upper,
                    lower_range,
                    upper_range,
                )
            else:
                raise ValueError

            processed_windows.append(processed_window)

        return processed_windows

    def feature_engineering(self, ss):
        lower_range = -1.0
        upper_range = 1.0

        processed_windows = []
        for signal_type in self.signal_types:
            processed_window = None

            if signal_type == "NP":
                np_decimate_one = 10
                np_decimate_two = 4
                processed_window = standardize(
                    decimate(
                        decimate(self.signal_dict["NP"][ss], np_decimate_one),
                        np_decimate_two,
                    )
                )
                skew = skewness(processed_window)
                kurt = kurtosis(processed_window)
                spectral_cent = spectral_centroid(processed_window, 5)
                spectral_sp = spectral_spread(processed_window, 5)
                spectral_sk = spectral_skewness(processed_window, 5)
                spectral_kt = spectral_kurtosis(processed_window, 5)
                features = [
                    skew,
                    kurt,
                    spectral_cent,
                    spectral_sp,
                    spectral_sk,
                    spectral_kt,
                ]
            elif signal_type == "Thorax":
                thorax_select = 4
                thorax_decimate = 10
                processed_window = standardize(
                    standardize(
                        decimate(
                            self.signal_dict["Thorax"][ss][::thorax_select],
                            thorax_decimate,
                        )
                    )
                    + standardize(
                        decimate(
                            self.signal_dict["Abdomen"][ss][::thorax_select],
                            thorax_decimate,
                        )
                    )
                )
                skew = skewness(processed_window)
                kurt = kurtosis(processed_window)
                spectral_cent = spectral_centroid(processed_window, 5)
                spectral_sp = spectral_spread(processed_window, 5)
                spectral_sk = spectral_skewness(processed_window, 5)
                spectral_kt = spectral_kurtosis(processed_window, 5)
                features = [
                    skew,
                    kurt,
                    spectral_cent,
                    spectral_sp,
                    spectral_sk,
                    spectral_kt,
                ]
            elif signal_type == "PR":
                pr_decimate_one = 10
                pr_decimate_two = 4
                processed_window = standardize(
                    decimate(
                        decimate(self.signal_dict["PR"][ss], pr_decimate_one),
                        pr_decimate_two,
                    )
                )
                skew = skewness(processed_window)
                kurt = kurtosis(processed_window)
                spectral_cent = spectral_centroid(processed_window, 5)
                spectral_sp = spectral_spread(processed_window, 5)
                spectral_sk = spectral_skewness(processed_window, 5)
                spectral_kt = spectral_kurtosis(processed_window, 5)
                features = [
                    skew,
                    kurt,
                    spectral_cent,
                    spectral_sp,
                    spectral_sk,
                    spectral_kt,
                ]
            elif signal_type == "HR":
                hr_select = 200
                hr_lower = 50.0
                hr_upper = 240.0
                processed_window = normalize_range(
                    self.signal_dict["HR"][ss][::hr_select],
                    hr_lower,
                    hr_upper,
                    lower_range,
                    upper_range,
                )
                first_moment = np.mean(processed_window)
                min_max = np.max(processed_window) - np.min(processed_window)
                features = [first_moment, min_max]
            elif signal_type == "SpO2":
                spo2_select = 100
                spo2_decimate = 2
                spo2_lower = 60.0
                spo2_upper = 100.0
                processed_window = normalize_range(
                    decimate(
                        self.signal_dict["SpO2"][ss][::spo2_select],
                        spo2_decimate,
                    ),
                    spo2_lower,
                    spo2_upper,
                    lower_range,
                    upper_range,
                )
                first_moment = np.mean(processed_window)
                min_max = np.max(processed_window) - np.min(processed_window)
                features = [first_moment, min_max]
            elif signal_type == "PCO2":
                pco2_select = 100
                pco2_decimate = 2
                pco2_lower = 30.0
                pco2_upper = 70.0
                processed_window = normalize_range(
                    decimate(
                        self.signal_dict["PCO2"][ss][::pco2_select],
                        pco2_decimate,
                    ),
                    pco2_lower,
                    pco2_upper,
                    lower_range,
                    upper_range,
                )
                first_moment = np.mean(processed_window)
                min_max = np.max(processed_window) - np.min(processed_window)
                features = [first_moment, min_max]
            else:
                raise ValueError
            processed_windows.extend(features)

        print(processed_windows)
        return np.array(processed_windows)


# NOTE: C1==C2 is not garantueed. Only in expectation.
class BalancedSampler(WeightedRandomSampler):
    def __init__(self, dataset, replacement=False):
        labels = dataset._get_labels()

        classes, counts = np.unique(labels, return_counts=True)
        weights = np.zeros_like(labels, dtype=float)
        for val, nums in zip(classes, counts):
            weights[labels == val] = 1.0 / nums
        num_samples = int(min(counts) * len(counts))

        super().__init__(
            weights=weights, num_samples=num_samples, replacement=replacement
        )

    def __iter__(self):
        return super().__iter__()

    def __len__(self):
        return super().__len__()


class MultiDatasetBalancedSampler(Sampler):
    def __init__(self, concat_dataset, replacement=False):
        self.concat_dataset = concat_dataset
        self.replacement = replacement

    def __iter__(self):
        offset = 0
        all_indices = []
        for dat in self.concat_dataset.datasets:
            indices = list(BalancedSampler(dat, replacement=self.replacement))
            offset_indicies = [idx + offset for idx in indices]
            all_indices.extend(offset_indicies)
            offset += len(dat)
        shuffle(all_indices)

        yield from iter(all_indices)

    def __len__(self):
        length = 0
        for dat in self.concat_dataset.datasets:
            length += len(BalancedSampler(dat, replacement=self.replacement))
        return length


def skewness(signal):
    return scipy.stats.skew(signal)


def kurtosis(signal):
    return scipy.stats.kurtosis(signal)


def calc_fft(signal, fs):
    fmag = np.abs(np.fft.rfft(signal))
    f = np.fft.rfftfreq(len(signal), d=1 / fs)

    return f.copy(), fmag.copy()


def spectral_centroid(signal, fs):
    f, fmag = calc_fft(signal, fs)
    if not np.sum(fmag):
        return 0
    else:
        return np.dot(f, fmag / np.sum(fmag))


def spectral_spread(signal, fs):
    f, fmag = calc_fft(signal, fs)
    spect_centroid = spectral_centroid(signal, fs)

    if not np.sum(fmag):
        return 0
    else:
        return np.dot(((f - spect_centroid) ** 2), (fmag / np.sum(fmag))) ** 0.5


def spectral_skewness(signal, fs):
    f, fmag = calc_fft(signal, fs)
    spect_centr = spectral_centroid(signal, fs)

    if not spectral_spread(signal, fs):
        return 0
    else:
        skew = ((f - spect_centr) ** 3) * (fmag / np.sum(fmag))
        return np.sum(skew) / (spectral_spread(signal, fs) ** 3)


def spectral_kurtosis(signal, fs):
    f, fmag = calc_fft(signal, fs)
    if not spectral_spread(signal, fs):
        return 0
    else:
        spect_kurt = ((f - spectral_centroid(signal, fs)) ** 4) * (fmag / np.sum(fmag))
        return np.sum(spect_kurt) / (spectral_spread(signal, fs) ** 4)
