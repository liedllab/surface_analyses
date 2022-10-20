from typing import List, Optional
import anarci
import bisect
from collections import namedtuple

CDRS = {
    'chothia': {
        'L': {
            1: slice(24, 35),
            2: slice(50, 57),
            3: slice(89, 98),
        },
        'H': {
            1: slice(26, 33),
            2: slice(52, 57),
            3: slice(95, 103),
        }
    }
}

# insertion_code defaults to ' ' to be consistent with ANARCI.
class ResidueCode(namedtuple('ResidueCode', ['index', 'insertion_code'], defaults=[' '])):
    def __init__(self, index, insertion_code=' '):
        if not isinstance(index, int):
            raise TypeError('index must be an integer')

class Chain:
    """
    Provides indexing functionality for a subset of the query residues.

    Stores a subset of the residues of the original query, as well as its own
    start index (offset)
    """
    def __init__(self, residues, offset):
        self.residues = residues
        self.offset = offset
        return

    def _integer_index(self, key: Optional[ResidueCode]):
        if key is None:
            return None
        return bisect.bisect_left(self.residues, key)

    @staticmethod
    def regularize_key(key):
        if isinstance(key, slice):
            start = None if key.start is None else Chain.to_residue_code(key.start)
            stop = None if key.stop is None else Chain.to_residue_code(key.stop)
            return slice(start, stop, key.step)
        else:
            return slice(Chain.to_residue_code(key), Chain.to_residue_code(key) + ('',))

    @staticmethod
    def to_residue_code(key):
        try:
            return ResidueCode(*key)
        except TypeError:  # not iterable
            return ResidueCode(key)

    def __getitem__(self, key):
        key = self.regularize_key(key)
        # "and" evaluates to 2nd operand if the 1st operand is False-like.
        # In this case, the start and stop elements are None or ResidueCode.
        start = self._integer_index(key.start)
        stop = self._integer_index(key.stop)
        max_range = range(self.offset, self.offset + len(self.residues))
        return max_range[start:stop:key.step]

    def __repr__(self):
        return f'{self.__class__.__name__}({self.residues}, offset={self.offset})'


class Annotation:
    def __init__(self, seq, run_name='ab', scheme='chothia'):
        self.seq = seq
        self.run_name = run_name
        self.scheme = scheme
        
        numbering, alignment_details, hit_tables = anarci.anarci([(run_name, seq)], scheme=scheme)
        self._chains = {}
        for chain, details in zip(numbering[0], alignment_details[0]):
            assert details['chain_type'] not in self._chains
            residues, start, stop = chain
            res_keys = [r[0] for r in residues if r[1] != '-']
            assert is_sorted(res_keys)
            # stop indicates the *last* index, so stop-start = len - 1
            # This is crucial since I use the position in res_keys to identify my residues!
            assert stop - start == len(res_keys) - 1, f"{start=}, {len(res_keys)=}, {stop=}"
            self._chains[details['chain_type']] = Chain(res_keys, start)

    @classmethod
    def from_traj(cls, traj, **kwargs):
        seq = get_sequence(traj.top)
        return cls(seq, **kwargs)

    def chain(self, name) -> Chain:
        try:
            return self._chains[name]
        except KeyError:
            if name == 'light':
                return self.light_chain()
            elif name == 'heavy':
                return self.heavy_chain()
            raise
    
    def light_chain(self) -> Chain:
        if 'K' in self._chains:
            return self._chains['K']
        return self._chains['L']
    
    def heavy_chain(self) -> Chain:
        return self._chains['H']

    def cdr_indices(self, cdr_def=None) -> List[int]:
        """Indices in the input sequence that correspond to the 6 CDR loops."""
        if cdr_def is None:
            cdr_def = CDRS[self.scheme]
        out = []
        heavy = self.chain('heavy')
        for loop in cdr_def['H'].values():
            for aa in heavy[loop]:
                out.append(aa)
        light = self.chain('light')
        for loop in cdr_def['L'].values():
            for aa in light[loop]:
                out.append(aa)
        return out

    def __repr__(self):
        return f'{self.__class__.__name__}({self.seq}, {self.run_name}, {self.scheme})'


def is_oneletter(r):
    if r is None:
        return False
    return len(r) == 1


def get_sequence(top):
    seq = []
    for r in top.residues:
        seq.append(r.code if is_oneletter(r.code) else 'X')
    assert len("".join(seq)) == top.n_residues
    return "".join(seq)

def is_sorted(lst):
    for i, element in enumerate(lst[1:]):
        if lst[i] > element:
            return False
    return True
