from concurrent.futures import ProcessPoolExecutor, as_completed
from numba import njit
import numpy as np
import scipy.cluster.hierarchy as hier
import scipy.spatial.distance as dist
import sgkit
import dask.distributed as dd
from dataclasses import dataclass

from ..transform import validate_spr
from ..transform import _subtree_prune, _subtree_regraft
from .felsenstein import update_temp_likelihoods_upward
from .felsenstein import clean_temp_nodes
from .felsenstein import update_likelihoods_upward


@dataclass
class LikelihoodTree:
    left_child: np.ndarray
    right_sib: np.ndarray
    node_parent: np.ndarray
    node_branch_length: np.ndarray
    node_likelihood: np.ndarray
    sample_node: np.ndarray
    locked_nodes: np.ndarray
    affected_node: np.ndarray
    affected_node_likelihood: np.ndarray
    rate: float
    pi: np.ndarray
    likelihood: float
    num_nodes: int


# def ts_to_dataset(ts, samples=None):
#     """
#     Convert the specified tskit tree sequence into an sgkit dataset.
#     Note this just generates haploids for now - see the note above
#     in simulate_ts.
#     """
#     chunks = None
#     if samples is None:
#         samples = ts.samples()
#     tables = ts.dump_tables()
#     alleles = []
#     genotypes = []
#     max_alleles = 0
#     for var in ts.variants(samples=samples):
#         alleles.append(var.alleles)
#         max_alleles = max(max_alleles, len(var.alleles))
#         genotypes.append(var.genotypes)
#     padded_alleles = [
#         list(site_alleles) + [""] * (max_alleles - len(site_alleles))
#         for site_alleles in alleles
#     ]
#     alleles = np.array(padded_alleles).astype("S")
#     genotypes = np.expand_dims(genotypes, axis=2)

#     ds = sgkit.create_genotype_call_dataset(
#         variant_contig_names=["1"],
#         variant_contig=np.zeros(len(tables.sites), dtype=int),
#         variant_position=tables.sites.position.astype(int),
#         variant_allele=alleles,
#         sample_id=np.array([f"tsk_{u}" for u in samples]).astype("U"),
#         call_genotype=genotypes,
#     )
#     if chunks is not None:
#         ds = ds.chunk({"variants": chunks})
#     return ds


# def simulate_ts(num_samples, sequence_length, mutation_rate, seed=1234):
#     tsa = msprime.sim_ancestry(
#         num_samples,
#         recombination_rate=0,
#         sequence_length=sequence_length,
#         ploidy=1,
#         random_seed=seed,
#     )
#     return msprime.sim_mutations(tsa, mutation_rate, random_seed=seed)


# def create_mutation_tree(ts_in, mutation_rate):
#     pk_mts = ts_to_dataset(ts_in)
#     ds_in = pk.from_tskit(ts_in.first())
#     ds = ds_in.merge(pk_mts)
#     ds.attrs["rate"] = mutation_rate
#     ds.attrs["pi"] = np.full((4, 4), 1 / 4)
#     ds["variant_allele"] = ("variants", "alleles"), pk.util.base_mapping(
#         ds.variant_allele.data, np.array([b"A", b"C", b"G", b"T"])
#     )
#     return ds


def _linkage_matrix_to_dataset(Z):
    n = Z.shape[0] + 1
    N = 2 * n
    parent = np.full(N, -1, dtype=np.int32)
    time = np.full(N, 0, dtype=np.float64)
    left_child = np.full(N, -1, dtype=np.int32)
    right_sib = np.full(N, -1, dtype=np.int32)
    for j, row in enumerate(Z):
        u = n + j
        time[u] = j + 1
        lc = int(row[0])
        rc = int(row[1])
        parent[lc] = u
        parent[rc] = u
        left_child[u] = lc
        right_sib[lc] = rc
    left_child[-1] = N - 2
    time[-1] = np.inf
    return parent, time, left_child, right_sib, n


def linkage_matrix_to_dataset(Z):
    """
    Scipy's hierarchy methods return a (n-1, 4) linkage matrix describing the clustering
    of the n observations. Each row in this matrix corresponds to an internal node
    in the tree.
    """
    parent, time, left_child, right_sib, n = _linkage_matrix_to_dataset(Z)
    return (
        parent,
        time,
        left_child,
        right_sib,
        np.arange(n),
    )


# This is a bad name - we should do something to group distance matrix
# based methods together and give a method argument to decide what the
# method is.
def upgma(ds):
    # TODO do something more sensible later with ploidy
    ds.sizes["ploidy"] == 1
    ds = ds.squeeze("ploidy")
    # TODO add option to keep this distance matrix, like we do in sgkit for expensive
    # intermediate calculations.
    # TODO not sure if this is doing anything sensible!
    D = sgkit.pairwise_distance(ds.call_genotype.T)
    D = D.compute()
    # Convert to condensed vector form.
    v = dist.squareform(D)
    # This performs UPGMA clustering
    Z = hier.average(v)
    parent, time, left_child, right_sib, samples = linkage_matrix_to_dataset(Z)
    ds.node_parent[:] = parent
    ds.node_time[:] = time
    ds.node_left_child[:] = left_child
    ds.node_right_sib[:] = right_sib
    ds.sample_node[:] = samples
    return ds


@njit
def safe_random_int(low, high):
    return np.random.randint(low, high)


@njit
def set_random_seed(seed):
    np.random.seed(seed)


@njit
def random_hill_climbing(
    left_child,
    right_sib,
    node_parent,
    branch_length,
    num_nodes,
    root,
    locked_nodes,
    node_variant_likelihood,
    rate,
    pi,
    current_max,
    affected_node,
    affected_node_likelihood,
    candidate_moves=10,
):
    max_value = current_max
    best_source, best_destination = -1, -1

    for _ in range(candidate_moves):
        source = safe_random_int(0, num_nodes)
        destination = safe_random_int(0, num_nodes)
        if validate_spr(
            source, destination, node_parent, num_nodes, root, locked_nodes
        ):
            reverse_spr, reverse_spr_branch_length = _subtree_prune(
                source,
                left_child,
                right_sib,
                node_parent,
                branch_length,
            )
            update_temp_likelihoods_upward(
                reverse_spr,
                left_child,
                right_sib,
                node_parent,
                branch_length,
                affected_node,
                affected_node_likelihood,
                node_variant_likelihood,
                rate,
                pi,
            )
            _subtree_regraft(
                source,
                destination,
                left_child,
                right_sib,
                node_parent,
                branch_length,
                -1,
            )
            update_temp_likelihoods_upward(
                source,
                left_child,
                right_sib,
                node_parent,
                branch_length,
                affected_node,
                affected_node_likelihood,
                node_variant_likelihood,
                rate,
                pi,
            )

            root_likelihood = affected_node_likelihood[root]
            # value = np.prod(np.sum(root_likelihood * 0.25, axis=1))
            value = np.sum(np.log(np.sum(root_likelihood * 0.25, axis=1)))

            clean_temp_nodes(
                source, node_parent, affected_node, affected_node_likelihood
            )
            clean_temp_nodes(
                reverse_spr, node_parent, affected_node, affected_node_likelihood
            )

            if value > max_value:
                max_value = value
                best_source, best_destination = source, destination

            _subtree_prune(source, left_child, right_sib, node_parent, branch_length)
            _subtree_regraft(
                source,
                reverse_spr,
                left_child,
                right_sib,
                node_parent,
                branch_length,
                reverse_spr_branch_length,
            )

    return max_value, best_source, best_destination


def thread_safe_MLE(
    likelihood_tree: LikelihoodTree, candidate_moves: int, num_iter: int, seed: int
):

    np.random.seed(seed)
    set_random_seed(seed)

    steps = []
    for iter in range(num_iter):
        max_value, best_source, best_destination = random_hill_climbing(
            likelihood_tree.left_child,
            likelihood_tree.right_sib,
            likelihood_tree.node_parent,
            likelihood_tree.node_branch_length,
            likelihood_tree.num_nodes,
            likelihood_tree.left_child[-1],
            likelihood_tree.locked_nodes,
            likelihood_tree.node_likelihood,
            likelihood_tree.rate,
            likelihood_tree.pi,
            likelihood_tree.likelihood,
            likelihood_tree.affected_node,
            likelihood_tree.affected_node_likelihood,
            candidate_moves,
        )
        if max_value <= likelihood_tree.likelihood:
            break
        likelihood_tree.likelihood = max_value

        reverse_spr, _ = _subtree_prune(
            best_source,
            likelihood_tree.left_child,
            likelihood_tree.right_sib,
            likelihood_tree.node_parent,
            likelihood_tree.node_branch_length,
        )

        update_likelihoods_upward(
            reverse_spr,
            likelihood_tree.left_child,
            likelihood_tree.right_sib,
            likelihood_tree.node_parent,
            likelihood_tree.node_branch_length,
            likelihood_tree.node_likelihood,
            likelihood_tree.rate,
            likelihood_tree.pi,
        )

        _subtree_regraft(
            best_source,
            best_destination,
            likelihood_tree.left_child,
            likelihood_tree.right_sib,
            likelihood_tree.node_parent,
            likelihood_tree.node_branch_length,
            -1,
        )

        update_likelihoods_upward(
            best_source,
            likelihood_tree.left_child,
            likelihood_tree.right_sib,
            likelihood_tree.node_parent,
            likelihood_tree.node_branch_length,
            likelihood_tree.node_likelihood,
            likelihood_tree.rate,
            likelihood_tree.pi,
        )
        steps.append((best_source, best_destination, max_value, iter))

    return likelihood_tree, steps


def to_likelihood_tree(ds):
    return LikelihoodTree(
        left_child=ds.node_left_child.data,
        right_sib=ds.node_right_sib.data,
        node_parent=ds.node_parent.data,
        node_branch_length=ds.node_branch_length.data,
        node_likelihood=ds.node_likelihood.data,
        sample_node=ds.sample_node.data,
        locked_nodes=ds.locked_nodes.data,
        affected_node=np.zeros(ds.nodes.shape[0], dtype=np.bool_),
        affected_node_likelihood=np.zeros_like(
            ds.node_likelihood.data, dtype=np.float64
        ),
        rate=ds.rate.data,
        pi=ds.pi.data,
        # likelihood=np.prod(
        #     np.sum(ds.node_likelihood.data[ds.node_left_child.data[-1]] * 0.25, axis=1)
        # ),
        likelihood=np.sum(
            np.log(
                np.sum(
                    ds.node_likelihood.data[ds.node_left_child.data[-1]] * 0.25,
                    axis=1,
                )
            )
        ),
        num_nodes=ds.nodes.shape[0],
    )


def execute_in_parallel(
    likelihood_tree: LikelihoodTree,
    num_moves=100,
    num_iter=1000,
    num_processes=4,
    jobs=100,
):

    results = []

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = []

        for _ in range(jobs):
            futures.append(
                executor.submit(
                    thread_safe_MLE,
                    likelihood_tree,
                    num_moves,
                    num_iter,
                    np.random.randint(0, 987654321),
                )
            )

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    return results


def dask_execute_in_parallel(
    likelihood_tree: LikelihoodTree,
    dask_client: dd.Client,
    seed=None,
    candidate_moves=100,
    num_iter=100,
    jobs=100,
):
    results = []
    futures = []

    if seed:
        np.random.seed(seed)
        set_random_seed(seed)

    for _ in range(jobs):
        futures.append(
            dask_client.submit(
                thread_safe_MLE,
                likelihood_tree,
                candidate_moves,
                num_iter,
                np.random.randint(0, 987654321),
            )
        )

    for future in dd.as_completed(futures):
        result = future.result()
        results.append(result)

    return results

