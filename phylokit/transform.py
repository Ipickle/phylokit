import numpy as np
import xarray

from . import core
from . import jit
from . import util


@jit.numba_njit()
def _permute_node_seq(nodes, ordering, reversed_map):
    ret = np.zeros_like(nodes, dtype=np.int32)
    for u, v in enumerate(ordering):
        old_node = nodes[v]
        if old_node != -1:
            ret[u] = reversed_map[old_node]
        else:
            ret[u] = -1
    return ret


def permute_tree(ds, ordering):
    """
    Returns a new dataset in which the tree nodes have been permuted according
    to the specified ordering such that node u in the new dataset will be
    equivalent to ``ordering[u]``.
    :param xarray.DataSet ds: The tree dataset to permute.
    :param list ordering: The permutation to apply to the nodes.
    :return: A new dataset with the permuted nodes.
    :rtype: xarray.DataSet
    """
    num_nodes = ds.node_left_child.shape[0]
    if len(ordering) != num_nodes:
        raise ValueError(
            "The length of the ordering must be equal to the number of nodes"
        )

    for node in ordering:
        util.check_node_bounds(ds, node)

    reversed_map = np.zeros(num_nodes, dtype=np.int32)
    for u, v in enumerate(ordering):
        reversed_map[v] = u

    return core.create_tree_dataset(
        parent=_permute_node_seq(ds.node_parent.data, ordering, reversed_map),
        left_child=_permute_node_seq(ds.node_left_child.data, ordering, reversed_map),
        right_sib=_permute_node_seq(ds.node_right_sib.data, ordering, reversed_map),
        samples=np.array([reversed_map[s] for s in ds.sample_node.data]),
    )


@jit.numba_njit()
def _subtree_prune(
    source,
    left_child,
    right_sib,
    parent,
    branch_length,
):
    reverse_spr = -1
    src_parent = parent[source]
    src_parent_parent = parent[src_parent]

    if left_child[src_parent] == source:
        reverse_spr = right_sib[source]
    elif right_sib[left_child[src_parent]] == source:
        reverse_spr = left_child[src_parent]

    if left_child[src_parent_parent] == src_parent:
        if left_child[src_parent] == source:
            right_sib[source] = -1

        elif right_sib[left_child[src_parent]] == source:
            right_sib[reverse_spr] = -1

        left_child[src_parent_parent] = reverse_spr
        right_sib[reverse_spr] = right_sib[src_parent]
        parent[reverse_spr] = src_parent_parent
        right_sib[src_parent] = -1
    else:
        if left_child[src_parent] == source:
            right_sib[source] = -1

        elif right_sib[left_child[src_parent]] == source:
            right_sib[reverse_spr] = -1

        parent[reverse_spr] = src_parent_parent
        right_sib[left_child[src_parent_parent]] = reverse_spr

    reverse_spr_branch_length = branch_length[src_parent]
    branch_length[reverse_spr] += branch_length[src_parent]
    parent[src_parent] = -1
    branch_length[src_parent] = 0

    return reverse_spr, reverse_spr_branch_length


@jit.numba_njit()
def _subtree_regraft(
    source,
    destination,
    left_child,
    right_sib,
    parent,
    branch_length,
    src_parent_branch_length,
):
    src_parent = parent[source]
    dest_parent = parent[destination]

    if left_child[dest_parent] == destination:
        right_sib[src_parent] = right_sib[destination]
        left_child[src_parent] = destination
        parent[destination] = src_parent
        left_child[dest_parent] = src_parent
        parent[src_parent] = dest_parent
        right_sib[destination] = source

    elif right_sib[left_child[dest_parent]] == destination:
        right_sib[left_child[dest_parent]] = src_parent
        parent[src_parent] = dest_parent
        left_child[src_parent] = destination
        parent[destination] = src_parent
        right_sib[destination] = source

    dest_branch_length = branch_length[destination]

    if src_parent_branch_length == -1:
        src_parent_branch_length = dest_branch_length / 2

    branch_length[destination] = dest_branch_length - src_parent_branch_length
    branch_length[src_parent] = src_parent_branch_length


@jit.numba_njit()
def is_valid_destination(source, destination, node_parent):
    """
    Determines if the source node is on the same path as the destination node.

    :param int source: The source node.
    :param int destination: The destination node.
    :return: True if the source node is not on the same path as the destination node.
    """
    parent = node_parent
    u = parent[destination]
    while u != -1:
        if u == source:
            return False
        u = parent[u]
    return True


@jit.numba_njit()
def validate_spr(source, destination, node_parent, num_nodes, root, locked_nodes):
    """
    Validates if the Subtree Pruning and Regrafting (SPR) operation
    between source and destination is feasible.

    :param int source: The source node.
    :param int destination: The destination node.
    :return: True if the SPR operation is feasible.
    """
    src_parent = node_parent[source]
    src_parent_parent = node_parent[src_parent]
    dest_parent = node_parent[destination]
    virtual_root = node_parent[root]

    if not is_valid_destination(source, destination, node_parent):
        return False

    if (
        source == destination
        or source < 0
        or destination < 0
        or source >= num_nodes
        or destination >= num_nodes
        or destination == src_parent
        or src_parent == dest_parent
        or locked_nodes[source]
        or src_parent == root
        or src_parent_parent == virtual_root
        or dest_parent == virtual_root
        or destination == virtual_root
        or destination == root
    ):
        return False
    return True


def spr(ds, source, destination):
    """
    Perform a Subtree Prune and Regraft (SPR) operation on the tree dataset.

    :param xarray.DataSet ds: The tree dataset.
    :param int source: The node to prune.
    :param int destination: The node to regraft.
    :return: The tree dataset with the specified SPR operation applied.
    """

    parent = ds.node_parent.data
    left_child = ds.node_left_child.data
    right_sib = ds.node_right_sib.data
    branch_length = ds.node_branch_length.data
    locked_nodes = ds.locked_nodes.data

    virtual_root = left_child.shape[0] - 1
    root = left_child[virtual_root]

    if not is_valid_destination(source, destination, parent):
        raise ValueError("Source node is an ancestor of the destination node")

    if (
        source == destination
        or source < 0
        or destination < 0
        or source >= ds.nodes.shape[0]
        or destination >= ds.nodes.shape[0]
    ):
        raise ValueError("Invalid source or destination node")

    src_parent = parent[source]
    dest_parent = parent[destination]
    src_parent_parent = parent[src_parent]

    if (
        destination == src_parent
        or src_parent == dest_parent
        or locked_nodes[source]
        or src_parent == root
        or src_parent_parent == virtual_root
        or dest_parent == virtual_root
        or destination == virtual_root
        or destination == root
    ):
        raise ValueError("Invalid source or destination node")

    _subtree_prune(
        source,
        left_child,
        right_sib,
        parent,
        branch_length,
    )

    _subtree_regraft(
        source,
        destination,
        left_child,
        right_sib,
        parent,
        branch_length,
        -1,
    )

    return ds


def init_locked_nodes(ds):
    ds["locked_nodes"] = xarray.DataArray(
        np.zeros(ds.node_left_child.shape[0], dtype=bool),
        dims=("nodes",),
    )
    ds["locked_nodes"][-1] = True
    ds["locked_nodes"][ds.node_left_child[-1]] = True
    ds["locked_nodes"][ds.node_left_child[ds.node_left_child]] = True
    ds["locked_nodes"][ds.node_right_sib[ds.node_left_child[ds.node_left_child]]] = True

    return ds
