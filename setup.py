from setuptools import setup

setup(
    name='majortom',
    version='0.1',    
    description='Expandable Datasets for Earth Observation',
    url='https://github.com/ESA-PhiLab/Major-TOM',
    author='Alistair Francis and Mikolaj Czerkawski',
    author_email="mikolaj.czerkawski@esa.int",
    package_dir={"majortom":"MajorTom"},
    install_requires=[
      "torch>=1.10.0",
      "torchvision",
      "pandas",
      "geopandas",
      "numpy",
      "rasterio",
      "shapely",
      "tqdm",
      "pillow",
      "fsspec",
      "pyarrow",
      "matplotlib"
    ],
)
