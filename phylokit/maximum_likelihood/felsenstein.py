import numpy as np

from .. import jit
from .. import util


@jit.numba_njit()
def _transition_probability(i, j, t, mu, pi=None):
    """
    Transition probability from state i to j with branch length t
    under the Jukes-Cantor model with mutation rate mu.

    $P_{i j}(t)=e^{-u t} \\delta_{i j}+\\left(1-e^{-u t}\right) \\pi_j$

    :param i: int, initial state
    :param j: int, final state
    :param t: float, branch length
    :param mu: float, mutation rate
    :param pi: np.array, transition probability matrix
    with shape (4, 4)
    :return: float, transition probability
    """
    delta_ij = 1.0 if i == j else 0.0
    return np.exp(-mu * t) * delta_ij + (np.float64(1.0) - np.exp(-mu * t)) * pi[i][j]


def _naive_calculate_likelihood(
    node,
    likelihood,
    left_child,
    right_sib,
    node_branch_length,
    rate,
    pi,
):
    for start_state in range(4):
        left_child_prob = 0
        right_sib_prob = 0
        for end_state in range(4):
            left_child_prob += likelihood[left_child][
                end_state
            ] * _transition_probability(
                start_state,
                end_state,
                node_branch_length[left_child],
                rate,
                pi,
            )
            right_sib_prob += likelihood[right_sib][
                end_state
            ] * _transition_probability(
                start_state,
                end_state,
                node_branch_length[right_sib],
                rate,
                pi,
            )
        likelihood[node][start_state] = left_child_prob * right_sib_prob


def _naive_likelihood_felsenstein(
    node_left_child,
    node_right_sib,
    call_genotype,
    num_nodes,
    traversal_postorder,
    sample_nodes,
    variant_allele,
    node_branch_length,
    rate,
    pi,
):
    ret = np.zeros(call_genotype.shape[0], dtype=np.float64)

    for i in range(call_genotype.shape[0]):
        likelihood = np.zeros((num_nodes, 4), dtype=np.float64)
        for j, sample_node in enumerate(sample_nodes):
            if call_genotype[i, j] == -1:
                likelihood[sample_node] = 0.25
            else:
                likelihood[sample_node][variant_allele[i][call_genotype[i][j][0]]] = 1.0
        for node in traversal_postorder:
            if node in sample_nodes:
                continue
            else:
                left_child = node_left_child[node]
                right_sib = node_right_sib[left_child]
                _naive_calculate_likelihood(
                    node,
                    likelihood,
                    left_child,
                    right_sib,
                    node_branch_length,
                    rate,
                    pi,
                )

        ret[i] = np.sum(likelihood[traversal_postorder[-1]] * 0.25)

    return ret


def naive_likelihood_felsenstein(
    ds,
    rate,
    pi=None,
):
    """
    Basic implementation of the likelihood function for the
    Felsenstein Likelihood calculation using the pruning algorithm.

    :param ds: phylokit.DataSet, tree data
    :param rate: float, mutation rate
    :param GENOTYPE_ARRAY: list, genotype mapping,
    :param pi: list, stationary distribution of states
    :return: float, likelihood
    """
    GENOTYPE_ARRAY = np.array([b"A", b"C", b"G", b"T"], dtype="S")

    if pi is None:
        pi = np.full((4, 4), 0.25, dtype=np.float64)

    likelihoods = _naive_likelihood_felsenstein(
        ds.node_left_child.data,
        ds.node_right_sib.data,
        ds.call_genotype.data,
        ds.nodes.shape[0],
        ds.traversal_postorder.data,
        ds.sample_node.data,
        util.base_mapping(ds.variant_allele.data, GENOTYPE_ARRAY),
        ds.node_branch_length.data,
        rate,
        pi,
    )

    ret = np.prod(likelihoods)

    return ret

@jit.numba_njit()
def _transition_matrix(t, mu, pi):
    """
    Calculates the transition matrix for a given branch length,
    mutation rate, and base frequencies.

    :param float t: The branch length.
    :param float mu: The mutation rate.
    :param np.array pi: The base frequencies.
    :return: The transition matrix.
    """
    exp_term = np.exp(-mu * t)

    transition_matrix = (1 - exp_term) * pi

    np.fill_diagonal(transition_matrix, exp_term + (1 - exp_term) * pi)

    return transition_matrix


@jit.numba_njit()
def _calculate_likelihood(likelihood, branch_length, rate, pi):
    """
    Calculates likelihoods based on provided child likelihoods,
    branch length, mutation rate, and base frequencies.

    :param np.array likelihood: The likelihoods of the child nodes.
    :param float branch_length: The branch length.
    :param float rate: The mutation rate.
    :param np.array pi: The base frequencies.
    :return: The transition likelihood to the parent node for all variants.
    """
    transition_matrix = _transition_matrix(branch_length, rate, pi)
    return np.dot(likelihood, transition_matrix)


@jit.numba_njit()
def _likelihood_felsenstein(
    node_left_child,
    node_right_sib,
    node_branch_length,
    sample_node,
    sample_node_mask,
    traversal_postorder,
    variant_allele,
    call_genotype,
    rate,
    pi,
):
    """
    Calculates the likelihood of a tree using Felsenstein's algorithm.

    :param np.array traversal_postorder: The postorder traversal of the tree.
    :return: The likelihood of the tree.
    """
    node_variant_likelihood = np.zeros(
        (node_left_child.shape[0], call_genotype.shape[0], 4), dtype=np.float64
    )
    for j, sample_node in enumerate(sample_node):
        for i in range(call_genotype.shape[0]):
            likelihood = node_variant_likelihood[sample_node]
            if call_genotype[i, j] == -1:
                likelihood[i] = 0.25
            else:
                likelihood[i, variant_allele[i, call_genotype[i, j, 0]]] = 1.0

    for node in traversal_postorder:
        if not sample_node_mask[node]:
            u = node_left_child[node]
            probs = _calculate_likelihood(
                node_variant_likelihood[u],
                node_branch_length[u],
                rate,
                pi,
            )
            v = node_right_sib[u]
            while v != -1:
                probs *= _calculate_likelihood(
                    node_variant_likelihood[v],
                    node_branch_length[v],
                    rate,
                    pi,
                )
                v = node_right_sib[v]
            node_variant_likelihood[node] = probs
    return node_variant_likelihood


def likelihood_felsenstein(ds):

    if "sample_node_mask" not in ds:
        ds["sample_node_mask"] = ("nodes"), np.zeros(ds.nodes.shape[0], dtype=np.bool_)
        ds.sample_node_mask[ds.sample_node] = True

    return _likelihood_felsenstein(
        ds.node_left_child.data,
        ds.node_right_sib.data,
        ds.node_branch_length.data,
        ds.sample_node.data,
        ds.sample_node_mask.data,
        ds.traversal_postorder.data,
        ds.variant_allele.data,
        ds.call_genotype.data,
        ds.rate.data,
        ds.pi.data,
    )


@jit.numba_njit()
def update_temp_likelihoods_upward(
    node,
    node_left_child,
    node_right_sib,
    node_parent,
    node_branch_length,
    affected_node,
    affected_node_likelihood,
    node_variant_likelihood,
    rate,
    pi,
):
    """
    Updates the temporary affected node likelihoods upward from a specific node.

    :param int node: The node to update.
    """

    parent = node_parent[node]
    while parent != -1:
        affected_node[parent] = True
        u = node_left_child[parent]
        if affected_node[u]:
            likelihood = affected_node_likelihood[u]
        else:
            likelihood = node_variant_likelihood[u]
        affected_node_likelihood[parent] = _calculate_likelihood(
            likelihood,
            node_branch_length[u],
            rate,
            pi,
        )
        v = node_right_sib[u]
        while v != -1:
            if affected_node[v]:
                likelihood = affected_node_likelihood[v]
            else:
                likelihood = node_variant_likelihood[v]
            affected_node_likelihood[parent] *= _calculate_likelihood(
                likelihood,
                node_branch_length[v],
                rate,
                pi,
            )
            v = node_right_sib[v]

        parent = node_parent[parent]


@jit.numba_njit()
def clean_temp_nodes(node, node_parent, affected_node, affected_node_likelihood):
    """
    Cleans the temporary nodes upward from a specific node.

    :param int node: The node to clean.
    """
    u = node_parent[node]
    while u != -1:
        affected_node[u] = False
        affected_node_likelihood[u] = 0.0
        u = node_parent[u]


@jit.numba_njit()
def update_likelihoods_upward(
    node,
    node_left_child,
    node_right_sib,
    node_parent,
    node_branch_length,
    node_variant_likelihood,
    rate,
    pi,
):
    """
    Updates the node likelihoods upward from a specific node.

    :param int node: The node to update.
    """

    parent = node_parent[node]
    while parent != -1:
        u = node_left_child[parent]
        likelihood = node_variant_likelihood[u]
        probs = _calculate_likelihood(
            likelihood,
            node_branch_length[u],
            rate,
            pi,
        )
        v = node_right_sib[u]
        while v != -1:
            likelihood = node_variant_likelihood[v]
            probs *= _calculate_likelihood(
                likelihood,
                node_branch_length[v],
                rate,
                pi,
            )
            v = node_right_sib[v]

        node_variant_likelihood[parent] = probs

        parent = node_parent[parent]
