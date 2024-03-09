import os
import yaml
import csv
import json
import argparse

import ray

import numpy as np
import h5py

from peakdiff.crystfel_stream.data import StreamManager, StreamConfig, StreamPeakDiff, StreamPeakDiffConfig

from bragg_peak_fitter.utils                   import get_patch_list
from bragg_peak_fitter.engine                  import PeakFitter
from bragg_peak_fitter.modeling.pseudo_voigt2d import PseudoVoigt2D

from dataclasses import dataclass
from typing import List

from .h5writer import write_results_to_h5

@ray.remote
def process_sub_event(patch, max_nfev = 2000):
    fitting_result = PeakFitter.fit(patch, max_nfev = max_nfev)

    # [[[ Produce a label ]]]
    # Obtain the model profile...
    model_params  = fitting_result.params
    model_profile = PseudoVoigt2D(model_params)

    # Only focus on the pseudo voigt2d profile without the background correction...
    model_profile.includes_bg = False
    H, W = patch.shape[-2:]
    y = np.arange(0, H)
    x = np.arange(0, W)
    Y, X = np.meshgrid(y, x, indexing = 'ij')
    model_no_bg_val = model_profile(Y, X)

    # Re-align the model so that all values are positive...
    model_no_bg_val -= model_no_bg_val.min()

    # Generate a label by selecting pixels that are larger than the half max of the profile...
    threshold_level = 0.5
    model_no_bg_val_half_max = model_no_bg_val >= (model_no_bg_val.max() * threshold_level)

    return model_no_bg_val_half_max, fitting_result


@ray.remote
def process_event_batch(batch_idx, event_batch, max_concurrent_subtasks, shared_data_ref):
    '''
    Each event corresponds to a frame idx.

    shared_data_ref is a top-level object ref, so ray will auto de-reference it.
    '''
    # Unpack the shared data...
    identifier      = shared_data_ref['identifier'     ]
    basename_h5     = shared_data_ref['basename_h5'    ]
    dir_h5          = shared_data_ref['dir_h5'         ]
    max_nfev        = shared_data_ref['max_nfev'       ]
    sigma_cut       = shared_data_ref['sigma_cut'      ]
    redchi_cut      = shared_data_ref['redchi_cut'     ]
    win_size        = shared_data_ref['win_size'       ]
    stream_peakdiff = shared_data_ref['stream_peakdiff']
    pixel_map_cache = shared_data_ref['pixel_map_cache']

    results = []
    for event in event_batch:
        # [[[ LABELING ]]]
        # Obtain img and peaks...
        frame_idx             = event
        img                   = get_img(frame_idx, stream_peakdiff)
        psana_event_tuple     = get_psana_event_tuple(frame_idx, stream_peakdiff)
        good_peaks, bad_peaks = split_peaks_by_sigma(frame_idx, stream_peakdiff, sigma_cut = sigma_cut)

        # Get a patch list for this event...
        good_peaks = np.array(good_peaks)
        peaks_y    = good_peaks[:, 0]
        peaks_x    = good_peaks[:, 1]
        patch_list = get_patch_list(peaks_y, peaks_x, img, win_size)

        # Submit sub-events (peak fitting) elastically...
        # ...Submit the init few processing tasks
        sub_event_futures = []
        sub_event_results = []
        for _ in range(min(max_concurrent_subtasks, len(patch_list))):
            patch = patch_list.pop(0)
            future = process_sub_event.remote(patch, max_nfev)
            sub_event_futures.append(future)

        # ...Continuously submit new tasks once an old one is complete (elastic submission)
        while patch_list or sub_event_futures:
            # ...Wait for one batch to complete
            done_futures, sub_event_futures = ray.wait(sub_event_futures, num_returns = 1)
            complete_futures = ray.get(done_futures[0])
            sub_event_results.append(complete_futures)

            # ...More sub tasks to process???
            if patch_list:
                patch = patch_list.pop(0)
                future = process_sub_event.remote(patch, max_nfev)
                sub_event_futures.append(future)

        ## sub_event_futures = [ process_sub_event.remote(patch, max_nfev) for patch in patch_list ]
        ## sub_event_results = ray.get(sub_event_futures)

        # Unpack the results of all sub events...
        mask_label_sub_events, fitting_result_sub_events = [ list(row) for row in zip(*sub_event_results) ]

        # Prepare the mask...
        mask = np.zeros_like(img, dtype = bool)
        mask_patch_list = get_patch_list(peaks_y, peaks_x, mask, win_size)

        # Assign labeled patches to mask patches...
        for mask_label, mask_patch in zip(mask_label_sub_events, mask_patch_list):
            mask_patch[:] = mask_label[:]

        # [[[ AGGREGATE INFORMATION FOR EXPORT ]]]
        # Find out bad fit...
        bad_fit_context_list      = []
        bad_fit_init_values_list  = []
        bad_fit_final_values_list = []
        bad_fit_idx_list          = [ idx for idx, result in enumerate(fitting_result_sub_events) if result.redchi > redchi_cut ]
        for i in bad_fit_idx_list:
            context      = (peaks_y[i], peaks_x[i], win_size)
            init_values  = fitting_result_sub_events[i].init_values
            final_values = fitting_result_sub_events[i].params.valuesdict()

            bad_fit_context_list.append(context)
            bad_fit_init_values_list.append(init_values)
            bad_fit_final_values_list.append(final_values)

        # Aggregate information for this event for exporting...
        event_result = {
            'identifier'                : identifier,
            'image'                     : img,
            'mask'                      : mask,
            'good_peaks'                : good_peaks,
            'bad_peaks'                 : bad_peaks,
            'detector'                  : get_detector_info(frame_idx, stream_peakdiff),
            'crystals'                  : get_crystal_info(frame_idx, stream_peakdiff),
            'bad_fit_context_list'      : bad_fit_context_list,
            'bad_fit_init_values_list'  : bad_fit_init_values_list,
            'bad_fit_final_values_list' : bad_fit_final_values_list,
            'psana_event_tuple'         : psana_event_tuple,
            'pixel_map'                 : pixel_map_cache.get(frame_idx, None)
        }
        results.append(event_result)

    # Write results to HDF5...
    file_h5 = f"{basename_h5}.{batch_idx:04d}.hdf5"
    path_h5 = os.path.join(dir_h5, file_h5)
    write_results_to_h5(path_h5, results)

    return batch_idx


def get_img(frame_idx, stream_peakdiff):
    return stream_peakdiff.stream_manager.get_img(frame_idx)


def get_psana_event_tuple(frame_idx, stream_peakdiff):
    return stream_peakdiff.stream_manager.get_psana_event_tuple(frame_idx)


def split_peaks_by_sigma(frame_idx, stream_peakdiff, sigma_cut = 1):
    '''
    [(y, x), (y, x), ...]

    Good peaks are worth of labeling.
    '''
    return stream_peakdiff.stream_manager.split_peaks_by_sigma(frame_idx, sigma_cut)


def get_detector_info(frame_idx, stream_peakdiff):
    keys = ( 'photon_energy_eV',
             'average_camera_length', )
    metadata = stream_peakdiff.stream_manager.stream_data[frame_idx]['CHUNK_BLOCK']['metadata']
    return { k : metadata[k] for k in keys }


def get_crystal_info(frame_idx, stream_peakdiff):
    '''
    There could be multiple crystals.
    '''
    keys = ( 'astar',
             'bstar',
             'cstar',
             'lattice_type',
             'centering',
             'unique_axis', )
    metadata_list = [ crystal['metadata'] for crystal in stream_peakdiff.stream_manager.stream_data[frame_idx]['CHUNK_BLOCK']['crystal'] ]
    return [ { k : metadata[k] for k in keys } for metadata in metadata_list]


def cache_pixel_maps(stream_peakdiff):
    print('Caching pixel maps...')
    return stream_peakdiff.stream_manager.cache_pixel_maps()


@dataclass
class AppConfig:
    identifier             : str
    path_csv               : str
    batch_size             : int
    max_concurrent_tasks   : int
    max_concurrent_subtasks: int
    max_nfev               : int
    sigma_cut              : float
    redchi_cut             : float
    win_size               : int
    basename_h5            : str
    dir_h5                 : str
    path_peakdiff_config   : str

def main():
    parser = argparse.ArgumentParser(description='Run the PeakNet Dataset Writer.')
    parser.add_argument('--yaml', help='Path to the YAML configuration file')
    args = parser.parse_args()

    ray.init(address='auto')

    # ___/ CONFIG \___
    # Load config...
    config_path = args.yaml
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    app_config = AppConfig(**config_dict)

    identifier              = app_config.identifier
    path_csv                = app_config.path_csv
    batch_size              = app_config.batch_size
    max_concurrent_tasks    = app_config.max_concurrent_tasks
    max_concurrent_subtasks = app_config.max_concurrent_subtasks
    max_nfev                = app_config.max_nfev
    sigma_cut               = app_config.sigma_cut
    redchi_cut              = app_config.redchi_cut
    win_size                = app_config.win_size
    basename_h5             = app_config.basename_h5
    dir_h5                  = app_config.dir_h5
    path_peakdiff_config    = app_config.path_peakdiff_config

    # Events to process...
    frame_idx_list = []
    with open(path_csv, 'r') as fh:
        lines = csv.reader(fh)
        for line in lines:
            if line:
                frame_idx_list.append(int(line[0].strip()))

    # Remove redundancy in csv...
    frame_idx_list = sorted(list(set(frame_idx_list)))
    print(f"Processing {len(frame_idx_list)} frames.")

    # Retrieve data source from the peakdiff...
    with open(path_peakdiff_config, 'r') as file:
        peakdiff_config = yaml.safe_load(file)

    # For user-friendliness, always re-use cache...
    peakdiff_config['stream_config'].ignores_cache = False
    peakdiff_config.ignores_cache                  = False

    # Configure peakdiff...
    stream_config          = StreamConfig(**peakdiff_config['stream_config'])
    dir_output             = peakdiff_config['dir_output']
    stream_peakdiff_config = StreamPeakDiffConfig(stream_config = stream_config, dir_output = dir_output)
    stream_peakdiff        = StreamPeakDiff(stream_peakdiff_config)

    # Precompute pixel map for this stream...
    pixel_map_cache = cache_pixel_maps(stream_peakdiff)

    # Save shared data into a ray object store...
    shared_data = {
        'identifier'              : identifier,
        'basename_h5'             : basename_h5,
        'dir_h5'                  : dir_h5,
        'max_nfev'                : max_nfev,
        'sigma_cut'               : sigma_cut,
        'redchi_cut'              : redchi_cut,
        'win_size'                : win_size,
        'stream_peakdiff'         : stream_peakdiff,
        'pixel_map_cache'         : pixel_map_cache,
    }
    shared_data_ref = ray.put(shared_data)

    # ___/ RAY TASKS \___
    # Create batches of events (one event corresponds to one frame)...
    num_events    = len(frame_idx_list)
    event_batches = [frame_idx_list[i:min(i + batch_size, num_events)] for i in range(0, num_events, batch_size)]

    # [[[ Elastic task scheduling ]]]
    # Submit the init few batch processing tasks...
    process_batch_idx_list = []   # Python list as a queue (fetch data with .pop)
    batch_idx = 0
    for _ in range(min(max_concurrent_tasks, len(event_batches))):
        # Get a batch from the batch queue...
        batch = event_batches.pop(0)

        # Submit this batch...
        process_batch_idx_list.append(process_event_batch.remote(batch_idx, batch, max_concurrent_subtasks, shared_data_ref))

        # Update batch index...
        batch_idx += 1

    # Continuously submit new tasks once an old one is complete (elastic submission)...
    while event_batches or process_batch_idx_list:
        # Wait for one batch to complete...
        done_batch_idx_list, process_batch_idx_list = ray.wait(process_batch_idx_list, num_returns = 1)
        completed_batch_idx = ray.get(done_batch_idx_list[0])

        # If there are more batches???
        if event_batches:
            # Submit the batch now...
            batch = event_batches.pop(0)
            process_batch_idx_list.append(process_event_batch.remote(batch_idx, batch, max_concurrent_subtasks, shared_data_ref))
            batch_idx += 1

    ray.shutdown()


if __name__ == "__main__":
    main()
