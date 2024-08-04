import numpy as np
from numba import prange

from . import jit


@jit.numba_njit()
def _is_unary(postorder, left_child, right_sib):
    for u in postorder:
        v = left_child[u]
        num_children = 0
        while v != -1:
            num_children += 1
            v = right_sib[v]
        if num_children == 1:
            return True
    return False


def is_unary(ds):
    """
    Returns whether any of the nodes has unary number of children.

    :param xarray.DataSet ds: The tree dataset to check.
    :return : Whether the tree is unary.
    :rtype : bool
    """
    return _is_unary(
        ds.traversal_postorder.data,
        ds.node_left_child.data,
        ds.node_right_sib.data,
    )


def check_node_bounds(ds, *args):
    """
    Checks that the specified node is within the tree.

    :param xarray.DataArray ds: The tree to check.
    :param int `*args`: The node IDs to check.
    :raises ValueError: If any of the nodes are outside the tree.
    """
    num_nodes = ds.node_parent.data.shape[0] - 1
    for u in args:
        if u < 0 or u > num_nodes:
            raise ValueError(f"Node {u} is not in the tree")


@jit.numba_njit()
def _get_num_roots(left_child, right_sib):
    u = left_child[-1]
    num_roots = 0
    while u != -1:
        num_roots += 1
        u = right_sib[u]
    return num_roots


def get_num_roots(ds):
    """
    Returns the number of roots in the tree.

    :pram xarray.DataSet ds: The tree dataset.
    :return : The number of roots.
    :rtype : int
    """
    return _get_num_roots(ds.node_left_child.data, ds.node_right_sib.data)


@jit.numba_njit()
def _branch_length(parent, time, u):
    ret = 0
    p = parent[u]
    if p != -1:
        ret = time[p] - time[u]
    return ret


def branch_length(ds, u):
    """
    Returns the length of the branch (in units of time) joining the specified
    node to its parent.

    :param xarray.DataSet ds: The tree dataset.
    :param int u: The node ID.
    :return : The length of the branch.
    :rtype : float
    """
    check_node_bounds(ds, u)
    return _branch_length(ds.node_parent.data, ds.node_time.data, u)


@jit.numba_njit()
def _get_node_branch_length(parent, time):
    ret = np.zeros_like(parent, dtype=np.float64)
    for i in range(parent.shape[0]):
        ret[i] = _branch_length(parent, time, i)
    return ret


def get_node_branch_length(ds):
    """
    Returns the branch length of each node in the tree.

    :param xarray.DataSet ds: The tree dataset.
    :return: The branch length of each node.
    :rtype: numpy.ndarray
    """
    return _get_node_branch_length(ds.node_parent.data, ds.node_time.data)


@jit.numba_njit(parallel=True)
def base_mapping(base_matrix, mapper_matrix):
    """
    Convert the base matrix with the mapper matrix.

    For example:
        base_matrix = [[b'A', b'C'], [b'C', b'G']]
        mapper_matrix = [b'A', b'C', b'G', b'T']

        result_matrix = [[0, 1], [1, 2]]

    :param numpy.ndarray base_matrix: The base matrix to convert.
    :param numpy.ndarray mapping_matrix: The mapping matrix.
    :return: The converted matrix.
    """
    base_shape = base_matrix.shape
    base_matrix = base_matrix.flatten()
    result_matrix = np.zeros_like(base_matrix, dtype=np.int8)
    for i in prange(base_matrix.shape[0]):
        for j in range(mapper_matrix.shape[0]):
            if base_matrix[i] == mapper_matrix[j]:
                result_matrix[i] = j
                break
    return result_matrix.reshape(base_shape)


@jit.numba_njit()
def numba_random_seed(seed):
    """
    Set the random seed for numba.

    :param int seed: The random seed.
    """
    np.random.seed(seed)
