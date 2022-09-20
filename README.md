# Installing
Once all the dependencies are installed, `surface_analyses` can be installed
using `pip install .`.

However, there are some dependencies which are not on PyPI: `gisttools` is
automatically installed from GitHub. `anarci_wrapper` and `TMalign_wrapper` are
included as git submodules. You have to install them manually by going to the
respective subdirectories and doing `pip install .` in each.
