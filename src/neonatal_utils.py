import logging
import os
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from numba import jit
from pyedflib import highlevel
from scipy.signal import decimate
from torch.utils.data import Dataset

log = logging.getLogger(__name__)

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


# Read in list of raw signals.
def read_data(pat_id, file_path):
    """
    Align raw signal (.edf file) with annotation (.txt file) and return dictionary of signals.
    """

    log.info(f"Reading ID {pat_id}.")

    signals, signal_headers, header = highlevel.read_edf(
        os.path.join(file_path, f"signals{pat_id}.edf")
    )

    all_signals = {}

    for sig_header, sig in zip(signal_headers, signals):
        if sig_header["label"] in SIGNAL_TYPE.keys():
            key, freq = SIGNAL_TYPE[sig_header["label"]]
            all_signals[key] = np.repeat(sig, 200 // freq)
        else:
            # Log unkown signal headers.
            log.info("Ignore unknown Signal Header")
            log.info(f"##{ sig_header['label'] }##")
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


@jit
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


@jit
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
    """
    Standardize numpy array.
    """

    arr_mean = np.mean(arr)
    arr_std = np.std(arr)
    return (arr - arr_mean) / (arr_std if (arr_std > 0.0) else 1.0)


def normalize_range(signal, lower_source, upper_source, lower_target, upper_target):
    """
    Range normalize numpy array given lower and upper tresholds.
    """

    assert lower_source < upper_source
    assert lower_target < upper_target

    return (
        (upper_target - lower_target)
        * ((signal - lower_source) / (upper_source - lower_source))
    ) + lower_target


def create_time_windows(anti_adverse_events, cutter, window_size, away, lag):
    """
    Create classification time windows with the specified lag and distance from adverse events.
    """

    assert away >= window_size + lag

    increasing = sum_from_left(anti_adverse_events)
    decreasing = sum_from_right(anti_adverse_events)
    slice_ls = create_slices(cutter)

    slices_and_labels = []
    for ss in slice_ls[:-1]:
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
    """
    Dataset providing classification time windows to the classification model.
    """

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
        self.pat_id = pat_id
        self.dataset_mode = dataset_mode
        self.signal_types = signal_types
        self.adverse_events = adverse_events
        self.cutter_events = cutter_events
        self.time_window = time_window
        self.lag = lag
        self.away = away
        self.signal_dict = signal_dict

        # Adverse events on which distances are calulated on. These are the anti-adverse events!
        # That is, 1 - adverse_event_mask.
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
        self.time_window_df["sig"] = self.time_window_df.apply(
            lambda row: self.feature_extraction(row["slice"]), axis=1
        )

    # Get undersampled elements for training.
    def __getitem__(self, index):
        if self.dataset_mode == "stacked":
            return self._get_stacked(index)
        elif self.dataset_mode == "list":
            return self._get_list(index)
        else:
            raise ValueError

    def __len__(self):
        return len(self.time_window_df)

    # Needs to be implemented for BalancedSamlping
    def _get_labels(self):
        return self.time_window_df["label"].to_numpy()

    # Stacked mode requires that all signals have the same length.
    # Trivially satisfied if only one signal is used.
    def _get_stacked(self, index):
        index_row = self.time_window_df.iloc[index]
        x_s = torch.from_numpy(np.row_stack(index_row["sig"]).astype(np.float32))
        y_s = torch.Tensor([np.float32(index_row["label"])])
        return (x_s, y_s)

    def _get_stacked_verbose(self, index):
        x_s, y_s = self._get_stacked(index)
        ss = self.time_window_df.iloc[index]["slice"]
        return (x_s, y_s, ss)

    def _get_list(self, index):
        index_row = self.time_window_df.iloc[index]
        x_s = [
            torch.from_numpy(ind.astype(np.float32)).unsqueeze(0)
            for ind in index_row["sig"]
        ]
        y_s = torch.Tensor([np.float32(index_row["label"])])
        return (x_s, y_s)

    def _get_list_verbose(self, index):
        x_s, y_s = self._get_list(index)
        ss = self.time_window_df.iloc[index]["slice"]
        return (x_s, y_s, ss)

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
            elif signal_type == "TA":
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
                        self.signal_dict["SpO2"][ss][::spo2_select], spo2_decimate
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
                        self.signal_dict["PCO2"][ss][::pco2_select], pco2_decimate
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


if __name__ == "__main__":
    pass
