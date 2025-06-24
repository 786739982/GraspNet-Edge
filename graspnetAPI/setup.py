from distutils.core import setup
from setuptools import find_packages
from setuptools.command.install import install
import os

setup(
    name='graspnetAPI',
    version='1.2.11',
    description='graspnet API',
    author='Hao-Shu Fang, Chenxi Wang, Minghao Gou',
    author_email='gouminghao@gmail.com',
    url='https://graspnet.net',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scipy',
        'transforms3d',
        'open3d',
        'trimesh',
        'tqdm',
        'Pillow',
        'opencv-python',
        'pillow',
        'matplotlib',
        'pywavefront',
        'trimesh',
        'scikit-image',
        'autolab_core',
        'autolab-perception',
        'cvxopt',
        'dill',
        'h5py',
        'scikit-learn',
        'grasp_nms'
    ]
)
