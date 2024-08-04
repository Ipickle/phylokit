import numpy as np
import pytest

import phylokit as pk


def equals(tree1, tree2):
    return (
        tree1.node_parent.equals(tree2.node_parent)
        and tree1.node_left_child.equals(tree2.node_left_child)
        and tree1.node_right_sib.equals(tree2.node_right_sib)
        # and tree1.node_time.equals(tree2.node_time)
        # and tree1.node_branch_length.equals(tree2.node_branch_length)
        # and tree1.sample_node.equals(tree2.sample_node)
    )


class TestTreeRegraftingLeafNode:
    # Tree1
    # 2.00┊       6      ┊
    #     ┊     ┏━┻━┓    ┊
    # 1.00┊     4   5    ┊
    #     ┊    ┏┻┓ ┏┻┓   ┊
    # 0.00┊    0 1 2 3   ┊
    #
    #
    # Tree2
    # 2.00┊      6       ┊
    #     ┊    ┏━┻━┓     ┊
    # 1.50┊    ┃   4     ┊
    #     ┊    ┃  ┏┻━┓   ┊
    # 1.00┊    ┃  5  ┃   ┊
    #     ┊    ┃ ┏┻┓ ┃   ┊
    # 0.00┊    0 2 3 1   ┊
    #     0              1
    #
    #
    # Tree3
    #  2.00┊        6    ┊
    #      ┊      ┏━┻━┓  ┊
    #  1.50┊      5   ┃  ┊
    #      ┊    ┏━┻┓  ┃  ┊
    #  1.00┊    4  ┃  ┃  ┊
    #      ┊   ┏┻┓ ┃  ┃  ┊
    #  0.00┊   0 1 2  3  ┊
    #
    #
    # Tree4
    #  2.00┊      6      ┊
    #      ┊    ┏━┻━┓    ┊
    #  1.50┊    5   ┃    ┊
    #      ┊    ┃   ┃    ┊
    #  1.00┊    ┃   4    ┊
    #      ┊   ┏┻┓ ┏┻┓   ┊
    #  0.00┊   0 2 3 1   ┊
    #
    #
    # Tree5
    # 2.00┊      6       ┊
    #     ┊    ┏━┻━┓     ┊
    # 1.00┊    ┃   4     ┊
    #     ┊    ┃  ┏┻━┓   ┊
    # 0.50┊    ┃  5  ┃   ┊
    #     ┊    ┃ ┏┻┓ ┃   ┊
    # 0.00┊    0 3 2 1   ┊
    #     0              1

    def pk_tree1(self):
        return pk.core.create_tree_dataset(
            parent=np.array([4, 4, 5, 5, 6, 6, -1, -1], dtype=np.int32),
            time=np.array([0, 0, 0, 0, 1, 1, 2, np.inf], dtype=np.float64),
            left_child=np.array([-1, -1, -1, -1, 0, 2, 4, 6], dtype=np.int32),
            right_sib=np.array([1, -1, 3, -1, 5, -1, -1, -1], dtype=np.int32),
            samples=np.array([0, 1, 2, 3], dtype=np.int32),
        )

    def pk_tree2(self):
        return pk.core.create_tree_dataset(
            parent=np.array([6, 4, 5, 5, 6, 4, -1, -1], dtype=np.int32),
            time=np.array([0, 0, 0, 0, 1.5, 1, 2, np.inf], dtype=np.float64),
            left_child=np.array([-1, -1, -1, -1, 5, 2, 0, 6], dtype=np.int32),
            right_sib=np.array([4, -1, 3, -1, -1, 1, -1, -1], dtype=np.int32),
            samples=np.array([0, 1, 2, 3], dtype=np.int32),
        )

    def pk_tree3(self):
        return pk.core.create_tree_dataset(
            parent=np.array([4, 4, 5, 6, 5, 6, -1, -1], dtype=np.int32),
            time=np.array([0, 0, 0, 0, 1, 1.5, 2, np.inf], dtype=np.float64),
            left_child=np.array([-1, -1, -1, -1, 0, 4, 5, 6], dtype=np.int32),
            right_sib=np.array([1, -1, -1, -1, 2, 3, -1, -1], dtype=np.int32),
            samples=np.array([0, 1, 2, 3], dtype=np.int32),
        )

    def pk_tree4(self):
        return pk.core.create_tree_dataset(
            parent=np.array([5, 4, 5, 4, 6, 6, -1, -1], dtype=np.int32),
            time=np.array([0, 0, 0, 0, 1, 1.5, 2, np.inf], dtype=np.float64),
            left_child=np.array([-1, -1, -1, -1, 3, 0, 5, 6], dtype=np.int32),
            right_sib=np.array([2, -1, -1, 1, -1, 4, -1, -1], dtype=np.int32),
            samples=np.array([0, 1, 2, 3], dtype=np.int32),
        )

    def pk_tree5(self):
        return pk.core.create_tree_dataset(
            parent=np.array([6, 4, 5, 5, 6, 4, -1, -1], dtype=np.int32),
            time=np.array([0, 0, 0, 0, 1, 0.5, 2, np.inf], dtype=np.float64),
            left_child=np.array([-1, -1, -1, -1, 5, 3, 0, 6], dtype=np.int32),
            right_sib=np.array([4, -1, -1, 2, -1, 1, -1, -1], dtype=np.int32),
            samples=np.array([0, 1, 2, 3], dtype=np.int32),
        )

    def test_regrafting(self):
        tree1 = self.pk_tree1()
        tree2 = self.pk_tree2()
        tree3 = self.pk_tree3()
        tree4 = self.pk_tree4()
        tree5 = self.pk_tree5()

        tree2_regrafted = pk.spr(tree1.copy(deep=True), 1, 5)
        tree3_regrafted = pk.spr(tree1.copy(deep=True), 2, 4)
        tree4_regrafted = pk.spr(tree3_regrafted.copy(deep=True), 1, 3)
        tree5_regrafted = pk.spr(tree4_regrafted.copy(deep=True), 2, 3)

        assert equals(tree2, tree2_regrafted)
        assert equals(tree3, tree3_regrafted)
        assert equals(tree4, tree4_regrafted)
        assert equals(tree5, tree5_regrafted)

    def test_regraphting_error(self):
        tree = self.pk_tree1()

        # source node is the destination node
        with pytest.raises(ValueError):
            pk.spr(tree, 6, 6)

        # negative node
        with pytest.raises(ValueError):
            pk.spr(tree, -1, 4)

        with pytest.raises(ValueError):
            pk.spr(tree, 4, -1)

        # parent of source node is the destination node
        with pytest.raises(ValueError):
            pk.spr(tree, 2, 5)

        # node is not in the tree
        with pytest.raises(ValueError):
            pk.spr(tree, 999, 4)

        with pytest.raises(ValueError):
            pk.spr(tree, 4, 999)

        # source parent is root
        with pytest.raises(ValueError):
            pk.spr(tree, 4, 5)


class TestTreeLarge:
    # Tree1
    #  3.00┊        14           ┊
    #      ┊     ┏━━━┻━━━┓       ┊
    #  2.00┊    10      13       ┊
    #      ┊   ┏━┻━┓   ┏━┻━┓     ┊
    #  1.00┊   8   9  11  12     ┊
    #      ┊  ┏┻┓ ┏┻┓ ┏┻┓ ┏┻┓    ┊
    #  0.00┊  0 1 2 3 4 5 6 7    ┊
    #
    #
    # Tree2
    #  3.00┊           14        ┊
    #      ┊       ┏━━━━┻━━━┓    ┊
    #  2.00┊      10        ┃    ┊
    #      ┊    ┏━━┻━━┓     ┃    ┊
    #  1.50┊    ┃    13     ┃    ┊
    #      ┊    ┃   ┏━┻━┓   ┃    ┊
    #  1.00┊    8   9  12  11    ┊
    #      ┊   ┏┻┓ ┏┻┓ ┏┻┓ ┏┻┓   ┊
    #  0.00┊   0 1 2 3 6 7 4 5   ┊
    #
    #
    # Tree3
    #  3.00┊        14           ┊
    #      ┊    ┏━━━━┻━━━┓       ┊
    #  2.50┊    ┃       10       ┊
    #      ┊    ┃     ┏━━┻━━┓    ┊
    #  2.00┊    ┃    13     ┃    ┊
    #      ┊    ┃   ┏━┻━┓   ┃    ┊
    #  1.00┊    9  11  12   8    ┊
    #      ┊   ┏┻┓ ┏┻┓ ┏┻┓ ┏┻┓   ┊
    #  0.00┊   2 3 4 5 6 7 0 1   ┊
    #
    #
    # Tree4
    #  3.00┊        14           ┊
    #      ┊     ┏━━━┻━━━┓       ┊
    #  2.00┊    10      13       ┊
    #      ┊   ┏━┻━┓   ┏━┻━┓     ┊
    #  1.00┊   8   9  12  11     ┊
    #      ┊  ┏┻┓ ┏┻┓ ┏┻┓ ┏┻┓    ┊
    #  0.00┊  0 1 2 3 6 7 4 5    ┊
    #
    #
    # Tree6
    #  3.00┊             14         ┊
    #      ┊         ┏━━━━┻━━━┓     ┊
    #  2.00┊        10        ┃     ┊
    #      ┊      ┏━━┻━━┓     ┃     ┊
    #  1.50┊      13    ┃     ┃     ┊
    #      ┊    ┏━┻━┓   ┃     ┃     ┊
    #  1.00┊    8  12   9    11     ┊
    #      ┊   ┏┻┓ ┏┻┓ ┏┻┓   ┏┻┓    ┊
    #  0.00┊   0 1 6 7 2 3   4 5    ┊

    def pk_tree1(self):
        return pk.core.create_tree_dataset(
            parent=np.array(
                [8, 8, 9, 9, 11, 11, 12, 12, 10, 10, 14, 13, 13, 14, -1, -1],
                dtype=np.int32,
            ),
            time=np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 1, 2, 3, np.inf],
                dtype=np.float64,
            ),
            left_child=np.array(
                [-1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 8, 4, 6, 11, 10, 14],
                dtype=np.int32,
            ),
            right_sib=np.array(
                [1, -1, 3, -1, 5, -1, 7, -1, 9, -1, 13, 12, -1, -1, -1, -1],
                dtype=np.int32,
            ),
            samples=np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32),
        )

    def pk_tree2(self):
        return pk.core.create_tree_dataset(
            parent=np.array(
                [8, 8, 9, 9, 11, 11, 12, 12, 10, 13, 14, 14, 13, 10, -1, -1],
                dtype=np.int32,
            ),
            time=np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 1, 1.5, 3, np.inf],
                dtype=np.float64,
            ),
            left_child=np.array(
                [-1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 8, 4, 6, 9, 10, 14],
                dtype=np.int32,
            ),
            right_sib=np.array(
                [1, -1, 3, -1, 5, -1, 7, -1, 13, 12, 11, -1, -1, -1, -1, -1],
                dtype=np.int32,
            ),
            samples=np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32),
        )

    def pk_tree3(self):
        return pk.core.create_tree_dataset(
            parent=np.array(
                [8, 8, 9, 9, 11, 11, 12, 12, 10, 14, 14, 13, 13, 10, -1, -1],
                dtype=np.int32,
            ),
            time=np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2.5, 1, 1, 2, 3, np.inf],
                dtype=np.float64,
            ),
            left_child=np.array(
                [-1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 13, 4, 6, 11, 9, 14],
                dtype=np.int32,
            ),
            right_sib=np.array(
                [1, -1, 3, -1, 5, -1, 7, -1, -1, 10, -1, 12, -1, 8, -1, -1],
                dtype=np.int32,
            ),
            samples=np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32),
        )

    def pk_tree4(self):
        return pk.core.create_tree_dataset(
            parent=np.array(
                [8, 8, 9, 9, 11, 11, 12, 12, 10, 10, 14, 13, 13, 14, -1, -1],
                dtype=np.int32,
            ),
            time=np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 1, 2, 3, np.inf],
                dtype=np.float64,
            ),
            left_child=np.array(
                [-1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 8, 4, 6, 12, 10, 14],
                dtype=np.int32,
            ),
            right_sib=np.array(
                [1, -1, 3, -1, 5, -1, 7, -1, 9, -1, 13, -1, 11, -1, -1, -1],
                dtype=np.int32,
            ),
            samples=np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32),
        )

    def pk_tree6(self):
        return pk.core.create_tree_dataset(
            parent=np.array(
                [8, 8, 9, 9, 11, 11, 12, 12, 13, 10, 14, 14, 13, 10, -1, -1],
                dtype=np.int32,
            ),
            time=np.array(
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 1, 1, 1.5, 3, np.inf],
                dtype=np.float64,
            ),
            left_child=np.array(
                [-1, -1, -1, -1, -1, -1, -1, -1, 0, 2, 13, 4, 6, 8, 10, 14],
                dtype=np.int32,
            ),
            right_sib=np.array(
                [1, -1, 3, -1, 5, -1, 7, -1, 12, -1, 11, -1, -1, 9, -1, -1],
                dtype=np.int32,
            ),
            samples=np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.int32),
        )

    def test_regrafting(self):
        tree1 = self.pk_tree1()
        tree2 = self.pk_tree2()
        tree3 = self.pk_tree3()
        tree4 = self.pk_tree4()
        tree6 = self.pk_tree6()

        tree2_regrafted = pk.spr(tree1.copy(deep=True), 12, 9)
        reversed_tree2 = pk.spr(tree2.copy(deep=True), 12, 11)
        tree3_regrafted = pk.spr(tree1.copy(deep=True), 8, 13)
        tree4_regrafted = pk.spr(tree1.copy(deep=True), 11, 9)
        reversed_tree4 = pk.spr(tree4_regrafted.copy(deep=True), 11, 12)
        tree5_regrafted = pk.spr(tree1.copy(deep=True), 9, 13)
        reversed_tree5 = pk.spr(tree5_regrafted.copy(deep=True), 9, 8)
        tree6_regrafted = pk.spr(tree1.copy(deep=True), 12, 8)

        assert equals(tree2, tree2_regrafted)
        assert equals(tree1, reversed_tree2)
        assert equals(tree3, tree3_regrafted)
        assert equals(tree4, reversed_tree4)
        assert equals(tree1, reversed_tree5)
        assert equals(tree6, tree6_regrafted)
