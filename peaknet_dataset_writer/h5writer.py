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
- data, group
  - data_0, group
    - image, dataset
      - shape: (H, W)
    - label, dataset
      - shape: (H, W)
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
        # ___/ SHARED METADATA \___
        # Create the shared metadata group...
        shared_metadata = f.create_group('shared_metadata')

        # Create subgroups in the shared metadat groups...
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

            # Obtain unique identifier...
            # Enforce the following format (exp, run)
            uid = identifier
            if not uid in uid_tracker: uid_tracker[uid] = {
                "pixel_map" : None,
            }

            # Obtain shape of the image...
            H, W = image.shape

            # [[[ Data ]]]
            # Create a subgroup for each image to store its associated data...
            data_subgroup = data_group.create_group(f"data_{enum_idx:04d}")

            # Store the image...
            data_subgroup.create_dataset("image", data=image,
                                                  dtype  = 'float32',
                                                  chunks = (H, W),)
            # Store the label...
            data_subgroup.create_dataset("label", data=mask,
                                                  dtype       = 'int',
                                                  chunks      = (H, W),
                                                  compression = "gzip",)

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

            # [[[ Metadata ]]]
            # Create a subgroup for metadata...
            metadata_subgroup = data_subgroup.create_group("metadata")

            # Store unique identifier for an image...
            metadata_subgroup.create_dataset("identifier", data = identifier)

            # Store detector info...
            detector_subgroup = metadata_subgroup.create_group("detector")
            detector_subgroup.create_dataset("name", data = "")
            for k, v in detector.items():
                detector_subgroup.create_dataset(k, data = v)

            # Store pixel map...
            key = "pixel_map"
            if uid_tracker[uid][key] is None:
                pixel_map_data = -np.ones((H, W, 3), dtype = int)
                uid_tracker[uid][key] = pixel_map_data
                shared_pixel_maps.create_dataset(uid, data = pixel_map_data, compression = "gzip")

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
