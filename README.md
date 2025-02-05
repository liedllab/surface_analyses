# PEP-Patch 
[![DOI:10.1021/acs.jcim.3c01490](http://img.shields.io/badge/DOI-10.1021/acs.jcim.3c01490-B31B1B.svg)](https://doi.org/10.1021/acs.jcim.3c01490)

The electrostatic properties of proteins arise from the number and distribution of polar and charged residues. Electrostatic interactions in proteins play a critical role in numerous processes such as molecular recognition, protein solubility, viscosity, and antibody developability. Thus, characterizing and quantifying electrostatic properties of a protein are prerequisites for understanding these processes. Here, we present PEP-Patch, a tool to visualize and quantify the electrostatic potential on the protein surface in terms of surface patches, denoting separated areas of the surface with a common physical property. 

The tool's main uses are to generate a molecular surface, map a potential to this surface, and define patches, i.e. connected areas on the surface with all positive or all negative potential values. Currently, the tool supports APBS to directly calcualte electrostatic potentials, any user-supplied potential map, or a  mapping based on hydrophobicity scales.  


## Installing
Once all the dependencies are installed, `surface_analyses` can be installed
using `pip install .`.

### Optional dependencies
While most Python-based dependencies are installed by pip, some parts of `pep_patch_electrostatic` use additional software.

* APBS (https://github.com/Electrostatics/apbs) is used to create an electrostatic potential for `pep_patch_electrostatic` if none is provided by the user.
* ANARCI (https://github.com/oxpig/ANARCI) is needed to assign CDRs in `pep_patch_electrostatic`.

## Computing electrostatic surfaces
The `pep_patch_electrostatic` script can use APBS (https://github.com/Electrostatics/apbs) to compute an electrostatic potential.
It then creates a molecular surface and colors it by the potential, or searches
for positive and negative surface patches.

`pep_patch_electrostatic` is presented in "PEP-Patch: Electrostatics in Protein-Protein Recognition, Specificity and Antibody Developability".
To reproduce the values in the paper, use version 0.2.0 and the input examples provided in the Supporting Information of the application note.
In newer versions, the input now allows for loading MD trajectory data with the ```mdtraj``` library:
```
pep_patch_electrostatic PARM CRD --apbs_dir apbs --ply_out potential -o patches.csv
```
with PARM specyfing a ```mdtraj``` compatible topology file and CRD a coordinate file.
To use a pdb file as input, simply add it as both PARM and CRD to the input:
```
pep_patch_electrostatic input.pdb input.pdb --apbs_dir apbs --ply_out potential -o patches.csv
```
Additionally, you can use `--ply_cmap` to set the cmap for the (continuous) .ply file output, and `--pos_patch_cmap`
and `--neg_patch_cmap` to set the color map for the patch .ply files.

## Assigning hydrophobicity values
The `pep_patch_hydrophobic` script can use hydrophobicity scales to assign hydrophobicity
to proteins. Additionally, the `eisenberg` and `crippen` scales are
pre-defined. See Eisenberg and McLachlan (1986) or Wildman and Crippen (1999)
for references. Additionally, using `rdkit-crippen` will use a SMILES input to 
assign Crippen parameters using the RdKit (useful for small molecules).

## Output formats
Both the electrostatics and hydrophobicity script allow writing surfaces to .ply files, using the [plyfile](https://github.com/dranjan/python-plyfile) library. The surfaces is colored using either the patches (where each patch gets a different color), or using the raw data (using the electrostatic potential at the vertices, or the hydrohobic potential approach of [Heiden et al.](https://doi.org/10.1007/BF00124359)). The raw data will also be written to PLY properties, and the coordinate units are chosen to match Pymol (units of 10^(-8)m).

Additionally, the hydrophobicity script supports output in the numpy npz format, which is more efficient when writing many surfaces.

## Examples
Visualize the hydrophobic potential on a non-protein molecule:
```
pep_patch_hydrophobic PARM CRD --scale rdkit-crippen --smiles SMILES --out OUT.npz --ply_out $OUT.ply --potential --patches
```
Here, PARM is any file recognized by mdtraj as a topology with bonds (e.g., a PDB file with CONECT records), CRD is a structure file or trajecotry (e.g., the same PDB file or an XTC trajectory), and smiles is a SMILES string used to assign bond orders to the topology (use single quotes to avoid bash substitutions).
## Citation
```
@article{Hoerschinger2023,
author = {Hoerschinger, Valentin J. and Waibl, Franz and Pomarici, Nancy D. and Loeffler, Johannes R. and Deane, Charlotte M. and Georges, Guy and Kettenberger, Hubert and Fernández-Quintero, Monica L. and Liedl, Klaus R.},
title = {PEP-Patch: Electrostatics in Protein–Protein Recognition, Specificity, and Antibody Developability},
journal = {Journal of Chemical Information and Modeling},
volume = {63},
number = {22},
pages = {6964-6971},
year = {2023},
doi = {10.1021/acs.jcim.3c01490},
}
```
