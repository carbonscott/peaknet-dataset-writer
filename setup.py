import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="peaknet_dataset_writer",
    version="24.02.16",
    author="Cong Wang",
    author_email="wangimagine@gmail.com",
    description="PeakNet dataset writer.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/carbonscott/bragg-peak-fitter",
    keywords = ['SFX', 'X-ray', 'Dataset curation', 'Data engine', 'LCLS'],
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points = {
        'console_scripts' : [
            'peaknet-dataset-writer=peaknet_dataset_writer.zarr_app:main',
        ],
    },
    python_requires='>=3.6',
    include_package_data=True,
)
