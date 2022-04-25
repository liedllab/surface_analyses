import copy

import mdtraj as md

class AmberCompatibleTopology(md.Topology):

    @classmethod
    def from_topology(cls, other):
        "Adapted from Topology.copy"
        out = cls()
        for chain in other.chains:
            c = out.add_chain()
            for residue in chain.residues:
                r = out.add_residue(residue.name, c, residue.resSeq, residue.segment_id)
                for atom in residue.atoms:
                    out.add_atom(atom.name, atom.element, r, serial=atom.serial)

        for bond in other.bonds:
            a1, a2 = bond
            out.add_bond(a1, a2, type=bond.type, order=bond.order)

        return out

    def create_standard_bonds(self):
        """Create bonds based on the atom and residue names for all standard residue types.

        Adapted from Topology.create_standard_bonds.
        """
        # Load the standard bond defitions.
        dummy = md.Topology()
        dummy.create_standard_bonds()
        standardBonds = copy.deepcopy(dummy._standardBonds)
        standardBonds["CYX"] = standardBonds["CYS"]
        standardBonds["HID"] = standardBonds["HIS"]
        standardBonds["HIE"] = standardBonds["HIS"]
        standardBonds["HIP"] = standardBonds["HIS"]
        standardBonds["ACE"] = [('CH3', 'H1'), ('CH3', 'H2'), ('CH3', 'H3'), ('CH3', 'C'), ('C', 'O')]
        standardBonds["NME"] = [('CH3', 'HH31'), ('CH3', 'HH32'), ('CH3', 'HH33'), ('CH3', 'N'), ('H', 'N')]
        for chain in self._chains:
            # First build a map of atom names to atoms.

            atomMaps = []
            for residue in chain._residues:
                atomMap = {}
                atomMaps.append(atomMap)
                for atom in residue._atoms:
                    atomMap[atom.name] = atom

            # Loop over residues and construct bonds.

            for i in range(len(chain._residues)):
                name = chain._residues[i].name
                if name in standardBonds:
                    for bond in standardBonds[name]:
                        if bond[0].startswith('-') and i > 0:
                            fromResidue = i-1
                            fromAtom = bond[0][1:]
                        elif (bond[0].startswith('+')
                                and i < len(chain._residues)):
                            fromResidue = i+1
                            fromAtom = bond[0][1:]
                        else:
                            fromResidue = i
                            fromAtom = bond[0]
                        if bond[1].startswith('-') and i > 0:
                            toResidue = i-1
                            toAtom = bond[1][1:]
                        elif (bond[1].startswith('+')
                                and i < len(chain._residues)):
                            toResidue = i+1
                            toAtom = bond[1][1:]
                        else:
                            toResidue = i
                            toAtom = bond[1]
                        if (fromAtom in atomMaps[fromResidue]
                                and toAtom in atomMaps[toResidue]):
                            self.add_bond(atomMaps[fromResidue][fromAtom],
                                            atomMaps[toResidue][toAtom])
