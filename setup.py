from setuptools import setup

setup(
    name="Surface analyses",
    version="0.1",
    description="Hydrophobicity analyses based on SASA",
    author="Franz Waibl",
    author_email="franz.waibl@uibk.ac.at",
    packages=['surface_analyses'],
    include_package_data=True,
    zip_safe=False,
    setup_requires=['pytest_runner'],
    tests_require=['pytest'],
    py_modules=[
        "surface_analyses.commandline",
        "surface_analyses.structure",
        "surface_analyses.propensities",
        "surface_analyses.hydrophobic_potential",
        "surface_analyses.pdb",
    ],
    entry_points={
        'console_scripts': ['surfscore=surface_analyses.commandline:main'],
    },
)
