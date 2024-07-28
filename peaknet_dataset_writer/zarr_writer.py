"""
ZARR data model for PeakNet Dataset

- input (image), tensor
- target, tensor
- pixel_map, groups with soft link
- metadata (JSON attribute for each input)
"""


import numpy as np
import zarr
import json
from numcodecs import Blosc, JSON

def write_results_to_zarr(path_zarr, inputs):
    '''
    results:
        [(img, mask, good_peaks, bad_peaks, fitting_results),
         (img, mask, good_peaks, bad_peaks, fitting_results),
         ...
        ]
    '''
    with zarr.open(path_zarr, 'w') as f:
        # -- Set up compressor
        compressor = Blosc(cname='zstd', clevel=3, shuffle=Blosc.SHUFFLE)
        metadata_compressor = JSON()

        # -- SHARED METADATA
        # Create the shared metadata group...
        shared_metadata = f.create_group('shared_metadata')

        # Create subgroups in the shared metadata groups...
        shared_pixel_maps = shared_metadata.create_group('pixel_maps')

        # Create the data group...
        uid_tracker = {}
        for enum_idx, input in enumerate(inputs):
            # Unpack data...
            image                     = input['image'                    ]
            mask                      = input['mask'                     ]
            good_peaks                = input['good_peaks'               ]
            bad_peaks                 = input['bad_peaks'                ]
            bad_fit_context_list      = input['bad_fit_context_list'     ]
            bad_fit_init_values_list  = input['bad_fit_init_values_list' ]
            bad_fit_final_values_list = input['bad_fit_final_values_list']
            detector                  = input['detector'                 ]
            crystals                  = input['crystals'                 ]
            identifier                = input['identifier'               ]
            psana_event_tuple         = input['psana_event_tuple'        ]
            pixel_map                 = input['pixel_map'                ]

            # Unpack the psana event tuple...
            exp, run, psana_event_idx = psana_event_tuple

            # Obtain unique identifier...
            # Enforce the following format (exp, run)
            uid = identifier if exp is None else f"{exp}_r{run}"
            if not uid in uid_tracker: uid_tracker[uid] = {
                "pixel_map" : None,
            }

            # Obtain shape of the image...
            H, W = image.shape

            # -- IMAGE AND LABEL
            if enum_idx == 0:
                B = len(inputs)
                dataset_images = f.create_dataset(
                    'images',
                    shape      = (B, H, W),
                    chunks     = (1, H, W),
                    dtype      = 'float32',
                    compressor = compressor,
                )
                dataset_labels = f.create_dataset(
                    'labels',
                    shape      = (B, H, W),
                    chunks     = (1, H, W),
                    dtype      = 'int',
                    compressor = compressor,
                )
                metadata = f.create_dataset(
                    'metadata',
                    shape        =  (B, ),
                    dtype        =  object,
                    object_codec = metadata_compressor,
                )

            dataset_images[enum_idx] = image
            dataset_labels[enum_idx] = mask

            # -- METADATA
            # Conditional modification of metadata
            psana_event_identifier = f"{identifier}_{psana_event_idx:06d}" if exp is None else f"{exp}_r{run}_{psana_event_idx:06d}"
            metadata_items = dict(
                good_peaks                = good_peaks.tolist(),
                bad_peaks                 = bad_peaks.tolist(),
                bad_fit_context_list      = bad_fit_context_list,
                bad_fit_init_values_list  = bad_fit_init_values_list,
                bad_fit_final_values_list = bad_fit_final_values_list,
                detector                  = detector,
                crystals                  = crystals,
                identifier                = psana_event_identifier,
                psana_event_tuple         = psana_event_tuple,
            )

            # Store pixel map...
            key = "pixel_map"
            if uid_tracker[uid][key] is None:
                uid_tracker[uid][key] = pixel_map
                shared_pixel_maps.create_dataset(uid, data = pixel_map, compressor = compressor)
            metadata_items[key] = f"/shared_metadata/pixel_maps/{uid}"

            # Save metadata...
            metadata[enum_idx] = metadata_items

        print(f"Exporting {path_zarr} done.")
