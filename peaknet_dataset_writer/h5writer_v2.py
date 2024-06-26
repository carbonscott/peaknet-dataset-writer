"""
H5 format for PeakNet Dataset

hdf5
- shared_metadata, group
  - pixel_maps, group
    - pixel_map, link
      - shape: (H, W, 3)
      - Why 3?  It maps (y, x) to (x, y, z) in the detector space.
    - pixel_map_1, dataset
    - ...
- images, dataset
  - shape: (B, H, W)
- labels, dataset
  - shape: (B, H, W)
- data, group
  - data_0, group
    - image_index, attribute
    - label_index, attribute
    - good_peaks, dataset (or should I use groups???)
      - shape: (N, 2)
      - N means the number of peaks, it depends on the prerequisites.
    - bad_peaks, dataset
      - shape: (N, 2)
      - N means the number of peaks, it depends on the prerequisites.
    - bad_fit_init_values_list, group
      - 0
        - a
        - amp
        - b
        - c
        - cx
        - cy
        - eta
        - gamma_x
        - gamma_y
        - sigma_x
        - sigma_y
      - ...
    - bad_fit_final_values_list, group
      - 0
        - same as above
      - ...
    - bad_fit_context_list, dataset
    - metadata, group
      - identifier, dataset, e.g. mfx13016_0036
      - psana_event_idx, e.g. 14324
      - detector, group
        - name, dataset
        - photon_energy, dataset
        - average_camera_length, dataset
      - pixel_map, group
        - **link** to one of the shared pixel map
      - crystals, group
        - crystal_0, group
          - astar
          - bstar
          - centering
          - cstar
          - lattice_type
          - unique_axis
      - sample, dataset
  - data_1, group
  - ...
"""


import numpy as np
import h5py

def write_results_to_h5(path_h5, inputs):
    '''
    results:
        [(img, mask, good_peaks, bad_peaks, fitting_results),
         (img, mask, good_peaks, bad_peaks, fitting_results),
         ...
        ]
    '''
    with h5py.File(path_h5, 'w') as f:
        # -- SHARED METADATA
        # Create the shared metadata group...
        shared_metadata = f.create_group('shared_metadata')

        # Create subgroups in the shared metadata groups...
        shared_pixel_maps = shared_metadata.create_group('pixel_maps')

        # Create the data group...
        uid_tracker = {}
        data_group = f.create_group('data')
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
                dataset_images = f.create_dataset('images',
                                                  shape  = (B, H, W),
                                                  chunks = (1, H, W),
                                                  dtype  = 'float32')
                dataset_labels = f.create_dataset('labels',
                                                  shape       = (B, H, W),
                                                  chunks      = (1, H, W),
                                                  dtype       = 'int',
                                                  compression =  "gzip")

            dataset_images[enum_idx] = image
            dataset_labels[enum_idx] = mask


            # -- DATA
            # Create a subgroup for each image to store its associated data...
            data_subgroup = data_group.create_group(f"data_{enum_idx:04d}")

            # Store the image...
            data_subgroup.attrs['image_index'] = enum_idx
            data_subgroup.attrs['label_index'] = enum_idx

            # Store good peaks...
            data_subgroup.create_dataset("good_peaks", data = good_peaks)

            # Store bad peaks...
            data_subgroup.create_dataset("bad_peaks", data = bad_peaks)

            # Store bad fitting ...
            # ...Context
            data_subgroup.create_dataset("bad_fit_context_list", data = bad_fit_context_list)

            # ...Init values
            bad_fit_init_values_list_subgroup = data_subgroup.create_group("bad_fit_init_values_list")
            for item_idx, item_dict in enumerate(bad_fit_init_values_list):
                group = bad_fit_init_values_list_subgroup.create_group(f"{item_idx}")
                for k, v in item_dict.items():
                    group.create_dataset(k, data = v)

            # ...Final values
            bad_fit_final_values_list_subgroup = data_subgroup.create_group("bad_fit_final_values_list")
            for item_idx, item_dict in enumerate(bad_fit_final_values_list):
                group = bad_fit_final_values_list_subgroup.create_group(f"{item_idx}")
                for k, v in item_dict.items():
                    group.create_dataset(k, data = v)


            # -- METADATA
            # Create a subgroup for metadata...
            metadata_subgroup = data_subgroup.create_group("metadata")

            # Store unique identifier for the associated experimental run...
            metadata_subgroup.create_dataset("identifier", data = identifier)

            # Store unique psana_event_tuple...
            psana_event_identifier = f"{identifier}_{psana_event_idx:06d}" if exp is None else f"{exp}_r{run}_{psana_event_idx:06d}"
            metadata_subgroup.create_dataset("psana_event_identifier", data = psana_event_identifier)

            # Store detector info...
            detector_subgroup = metadata_subgroup.create_group("detector")
            detector_subgroup.create_dataset("name", data = "")
            for k, v in detector.items():
                detector_subgroup.create_dataset(k, data = v)

            # Store pixel map...
            key = "pixel_map"
            if uid_tracker[uid][key] is None:
            ##     pixel_map_data = -np.ones((H, W, 3), dtype = int)
                uid_tracker[uid][key] = pixel_map
                shared_pixel_maps.create_dataset(uid, data = pixel_map, compression = "gzip")

            metadata_subgroup[key] = h5py.SoftLink(f"/shared_metadata/pixel_maps/{uid}")

            # Store unit cell...
            crystals_subgroup = metadata_subgroup.create_group("crystals")
            for crystal_idx, crystal in enumerate(crystals):
                crystal_subgroup = crystals_subgroup.create_group(f"{crystal_idx}")
                for k, v in crystal.items():
                    crystal_subgroup.create_dataset(k, data = v)

            # Store sample...
            sample = metadata_subgroup.create_dataset("sample", data = "")
        print(f"Exporting {path_h5} done.")
