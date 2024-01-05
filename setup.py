from setuptools import setup, find_packages

setup(
    name="Surface analyses",
    version="0.1",
    description="Hydrophobicity analyses based on SASA",
    author="Franz Waibl",
    author_email="franz.waibl@uibk.ac.at",
    packages=["surface_analyses", "surface_analyses.anarci_wrapper"],
    include_package_data=True,
    zip_safe=False,
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'scikit-image',
        'gisttools @ git+https://github.com/liedllab/gisttools.git',
        'msms_wrapper @ git+https://github.com/rinikerlab/msms_wrapper',
        'plyfile',
        'matplotlib>=3.7',
        'pdb2pqr>=3',
        'biopython',
        'rdkit'],
    setup_requires=['pytest_runner'],
    tests_require=['pytest'],
    py_modules=[
        "surface_analyses.commandline_hydrophobic",
        "surface_analyses.structure",
        "surface_analyses.propensities",
        "surface_analyses.hydrophobic_potential",
        "surface_analyses.pdb",
        "surface_analyses.commandline_electrostatic",
    ],
    entry_points={
        'console_scripts': [
            'pep_patch_hydrophobic=surface_analyses.commandline_hydrophobic:main',
            'pep_patch_electrostatic=surface_analyses.commandline_electrostatic:main',
        ],
    },
)
