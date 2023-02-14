# Installing
Once all the dependencies are installed, `surface_analyses` can be installed
using `pip install .`.

However, there are some dependencies which are not on PyPI: `gisttools` is
automatically installed from GitHub. `anarci_wrapper` and `TMalign_wrapper` are
included as git submodules. You have to install them manually using pip.

Installation might look like this:

```pip install anarci_wrapper/
pip install TMalign_wrapper/
pip install .
```

# Assigning hydrophobicity values
The `surfscore` script can use hydrophobicity scales to assign hydrophobicity
to proteins. Additionally, the `eisenberg` and `crippen` scales are
pre-defined. See Eisenberg and McLachlan (1986) or Wildman and Crippen (1999)
for references.

For non-proteins, RdKit can be used to assign Wildman and Crippen parameters.
To do so, specify `--scale rdkit-crippen` and supply a mol2 additionally to the
normal input. The mol2 file must have correct bond orders and the same order of
atoms as the topology! If no such mol2 file is available, you can use the
`reorder_mol2.sh` script in the `extra-scripts` subdirectory.

# Examples
Visualize the hydrophobic potential on a non-protein molecule, using an
*ordered* mol2 file as reference:
```surfscore PARM CRD --scale rdkit-crippen --mol2 ORDERED_MOL2FILE --out OUT.npz --ply_out $OUT.ply --potential --patches
```
