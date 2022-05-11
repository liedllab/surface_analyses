import re
from bisect import bisect_right
from collections import namedtuple, defaultdict
from itertools import zip_longest, chain

from .pdb import AbstractAtom

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks. From https://docs.python.org/3/library/itertools.html#recipes"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

FORMAT_RE_PATTERN = re.compile(r"%FORMAT\(([0-9]+)([a-zA-Z]+)([0-9]+)\.?([0-9]*)\)")

class RawTopology:
    """Manages an AMBER prmtop topology file from disk, as a dict of lists, 
    where the keys are the flag names. The data is converted to the type 
    indicated by %FORMAT.
    
    A RawTopology is meant to be immutable. This is not strictly enforced, but
    some methods rely on it!
    
    Attributes:
    * data: dict of flag: list of entries
    * formats: dict of flag: Format
    * version_info: str
    """
    
    def __init__(self, data, formats, version_info):
        self._data = data
        self._formats = formats
        self.version_info = version_info
        bonds_with_h = data.get('BONDS_INC_HYDROGEN', [])
        bonds_without_h = data.get('BONDS_WITHOUT_HYDROGEN', [])
        self.bonds_inc_hydrogen = self.get_bonds_dict(bonds_with_h)
        self.bonds_without_hydrogen = self.get_bonds_dict(bonds_without_h)
        self.all_bonds = self.get_bonds_dict(chain(bonds_with_h, bonds_without_h))
        return

    @property
    def data(self):
        return self._data

    @property
    def formats(self):
        return self._formats

    def iter_atoms(self, nmax=None):
        if nmax is None:
            nmax = self.n_atoms()
        n = min(nmax, self.n_atoms())
        for i in range(n):
            yield self.atom(i)

    def atom(self, i):
        if i >= self.n_atoms():
            raise IndexError(f'index {i} is out of bounds with {self.n_atoms()} atoms.')
        return ParmAtom(i, self)

    @classmethod
    def from_file_name(cls, filename):
        """Load an AMBER prmtop topology file from disk, as a dict of lists, 
        where the keys are the flag names. The data is converted to the type 
        indicated by %FORMAT.
        """
        with open(filename) as f:
            return cls.from_file_handle(f)

    @classmethod
    def from_file_handle(cls, f):
        """Load an AMBER prmtop topology file from disk, as a dict of lists, 
        where the keys are the flag names. The data is converted to the type 
        indicated by %FORMAT.

        Parameters
        ----------
        f : file handle
        
        Notes
        -----
        Adapted from mdtraj.
        """

        version_info = None
        formats = {}
        data = {}
        ignoring = False

        for line in f:
            if line[0] == '%':
                if line.startswith('%VERSION'):
                    _, version_info = line.rstrip().split(None, 1)

                elif line.startswith('%FLAG'):
                    _, flag = line.rstrip().split(None, 1)
                    data[flag] = []
                    ignoring = flag in ('TITLE', 'CTITLE')

                elif line.startswith('%FORMAT'):
                    formats[flag] = Format.from_string(line)
                    
                elif line.startswith('%COMMENT'):
                    continue

            elif not ignoring:
                fmt = formats[flag]
                line = line.rstrip()
                for item in fmt.split_line(line):
                    data[flag].append(fmt.type(item))

        return cls(data, formats, version_info)
    
    def n_protein_atoms(self):
        "Number of atoms in non-solvent molecules, according to SOLVENT_POINTERS."
        n_mols = self.data['SOLVENT_POINTERS'][2] - 1
        return sum(self.data['ATOMS_PER_MOLECULE'][:n_mols])

    def n_atoms(self):
        "Number of atoms in the topology, using ATOM_NAME."
        return len(self.data['ATOM_NAME'])

    def residue_ids(self, n=None):
        """Residue id for each atom in the topology.
        
        Optionally stops after n atoms.
        """
        for at in self.iter_atoms(n):
            yield at.residue_id

    def get_bonds_dict(self, flat_bonds):
        """Return a dict with lists of bonded atoms for each atom. 
        
        Notes
        -----
        The total number of entries returned will be 2*N_BONDS, since every
        bond is assigned to both involved atoms.
        """
        out = defaultdict(list)
        # Bonds in an amber topology are stored as a flat list of 3*N_BONDS 
        # entries. 3 numbers correspond to one bond:
        # 1. the index of the first atom multiplied by 3,
        # 2. the index of the second atom multiplied by 3,
        # 3. the index of the bond type.
        for a, b, _ in grouper(flat_bonds, 3):
            out[a//3].append(b//3)
            out[b//3].append(a//3)
        return dict(out)

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self.data)} flags, {self.version_info})"


class Format(namedtuple('Format', ['num', 'type', 'length', 'precision'])):
    """
    A typical AMBER format string looks like: %FORMAT(5E16.8)
    
    * 5 is the number of fields per line
    * E is the data type (E: float, I: int, a: string)
    * 16 is the length of the field
    * 8 is the precision for floats. With I and a types, the final .8 is left out.
    """
    @classmethod
    def from_string(cls, s):
        """
        >>> Format.from_string('%FORMAT(5E16.8)')
        Format(num=5, type=<class 'float'>, length=16, precision=8)
        """
        m = FORMAT_RE_PATTERN.search(s.strip())
        known_types = {'a': str, 'I': int, 'E': float}
        if m is None:
            raise ValueError('Unknown format string: {:s}'.format(s))
        else:
            num = int(m.group(1))
            type = known_types.get(m.group(2), str)
            length = int(m.group(3))
            precision = int(m.group(4)) if m.group(4) != '' else 0
            return cls(num, type, length, precision)

    def split_line(self, line):
        """Yield segments of line according to self.length, and strip them.
        
        No type conversion is done here, and iteration stops when the line is too short. 
        When the line length is sufficient, but there are only spaces left, yield an empty string.
        
        Examples
        --------
        >>> list(Format(20, int, 4, 0).split_line('3   2   1   0'))
        ['3', '2', '1', '0']
        >>> list(Format(20, int, 4, 0).split_line('3   2        '))
        ['3', '2', '', '']
        """
        for index in range(0, len(line), self.length):
            item = line[index:index+self.length]
            if item:
                yield item.strip()


class ParmAtom(namedtuple('ParmAtom', ['i', 'top']), AbstractAtom):

    @property
    def residue_id(self):
        res_ptr = self.top.data['RESIDUE_POINTER']
        return bisect_right(res_ptr, self.i + 1) - 1

    @property
    def residue_label(self):
        return self.top.data['RESIDUE_LABEL'][self.residue_id]

    @property
    def name(self):
        return self.top.data['ATOM_NAME'][self.i]

    @property
    def atomic_number(self):
        return self.top.data['ATOMIC_NUMBER'][self.i]

    @property
    def atom_type(self):
        return self.top.data['AMBER_ATOM_TYPE'][self.i]

    @property
    def bonded_atoms(self):
        return [self.__class__(i, self.top) for i in self.top.all_bonds[self.i]]

    def __repr__(self):
        return f'ParmAtom {self.i} of {self.top}'
