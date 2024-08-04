import msprime
import numpy as np
import pytest
import tskit

import phylokit as pk


class TestIsUnary:

    def tree_balance(self):
        return pk.from_tskit(tskit.Tree.generate_balanced(4))

    def tree_unary(self):
        return pk.core.create_tree_dataset(
            parent=np.array([3, 4, 4, 5, 5, -1, -1]),
            left_child=np.array([-1, -1, -1, 0, 1, 3, 5]),
            right_sib=np.array([-1, 2, -1, 4, -1, -1, -1]),
            samples=np.array([0, 1, 2]),
        )

    def test_balance(self):
        assert pk.is_unary(self.tree_balance()) == False

    def test_unary(self):
        assert pk.is_unary(self.tree_unary()) == True


class TestCheckNodeBounds:

    def tree(self):
        return pk.from_tskit(tskit.Tree.generate_balanced(4))

    def test_in_bounds(self):
        pk.check_node_bounds(self.tree(), 0, 1, 2, 3)

    def test_out_of_bounds(self):
        with pytest.raises(ValueError):
            pk.check_node_bounds(self.tree(), 10)


class TestGetNumRoots:

    def tree(self):
        return pk.from_tskit(tskit.Tree.generate_balanced(4))

    def tsk_tree_mutiroots(self):
        tables = tskit.Tree.generate_balanced(9, arity=2).tree_sequence.dump_tables()
        edges = tables.edges.copy()
        tables.edges.clear()
        for edge in edges:
            if edge.parent != 16:
                tables.edges.append(edge)
        return pk.from_tskit(tables.tree_sequence().first())

    def test_num_roots(self):
        assert pk.get_num_roots(self.tree()) == 1

    def test_num_roots_multi(self):
        assert pk.get_num_roots(self.tsk_tree_mutiroots()) == 2


class TestBranchLength:

    def tree(self):
        msp_tree = msprime.simulate(10, random_seed=1)
        return msp_tree.first()

    def test_branch_length(self):
        tree = self.tree()
        pk_tree = pk.from_tskit(tree)

        for u in range(pk_tree.node_parent.data.shape[0] - 1):
            assert pk.get_node_branch_length(u) == tree.branch_length(u)
