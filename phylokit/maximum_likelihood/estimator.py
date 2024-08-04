import numpy as np
from numba import boolean
from numba import float64
from numba import int32
from numba import int8
from numba.experimental import jitclass

from ..traversal import _postorder

spec = [
    ("node_parent", int32[:]),
    ("node_left_child", int32[:]),
    ("node_right_sib", int32[:]),
    ("node_time", float64[:]),
    ("node_branch_length", float64[:]),
    ("sample_node", int32[:]),
    ("sample_node_mask", boolean[:]),
    ("node_likelihood", float64[:, :]),
    ("num_nodes", int32),
    ("call_genotype", int32[:, :, :]),
    ("variant_allele", int8[:, :]),
    ("locked_nodes", boolean[:]),
    ("rate", float64),
    ("node_variant_likelihood", float64[:, :, :]),
    ("pi", float64[:, :]),
    ("affected_node", boolean[:]),
    ("affected_node_likelihood", float64[:, :, :]),
    ("log_scale", boolean),
]


@jitclass(spec)
class PKMLEstimator:
    """
    PhyloKit Maximum Likelihood Estimator (PKMLEstimator)
    A class to estimate the maximum likelihood of a phylogenetic tree
    using a variety of computational methods optimized with Numba.
    """

    def __init__(
        self,
        node_parent,
        node_left_child,
        node_right_sib,
        node_time,
        node_branch_length,
        sample_node,
        call_genotype,
        variant_allele,
        rate,
        pi,
        log_scale=False,
    ):
        """
        Initializes the PKMLEstimator with tree structure and genetic data.

        :param np.array node_parent: The parent nodes of the tree.
        :param np.array node_left_child: The left child nodes of the tree.
        :param np.array node_right_sib: The right sibling nodes of the tree.
        :param np.array node_time: The time of the nodes.
        :param np.array node_branch_length: The branch length of the nodes.
        :param np.array sample_node: The sample nodes of the tree.
        :param np.array call_genotype: The genotype calls of the samples.
        :param np.array variant_allele: The variant alleles of the samples.
        :param float rate: The mutation rate.
        :param np.array pi: The base frequencies.
        :param bool log_scale: Whether to use logarithmic scaling.
        """
        self.log_scale = log_scale
        self.node_parent = node_parent.copy()
        self.node_left_child = node_left_child.copy()
        self.node_right_sib = node_right_sib.copy()
        self.node_time = node_time.copy()
        self.node_branch_length = node_branch_length.copy()
        self.sample_node = sample_node.copy()
        self.num_nodes = node_parent.shape[0]
        self.sample_node_mask = self.init_sample_node_mask()
        self.call_genotype = call_genotype.copy()
        self.variant_allele = variant_allele.copy()
        self.rate = rate
        self.pi = pi
        self.locked_nodes = self.init_locked_nodes()
        self.affected_node = np.zeros(self.num_nodes, dtype=np.bool_)
        self.init_likelihood()

    def init_locked_nodes(self):
        """
        Initialize nodes that should not be moved during optimization.
        """
        virtual_root = self.num_nodes - 1
        root = self.node_left_child[virtual_root]
        locked_nodes = np.zeros(self.num_nodes, dtype=np.bool_)
        locked_nodes[virtual_root] = True
        locked_nodes[root] = True
        locked_nodes[self.node_left_child[root]] = True
        locked_nodes[self.node_right_sib[self.node_left_child[root]]] = True
        return locked_nodes

    def init_likelihood(self):
        """
        Initialize the likelihood matrices depending on whether
        logarithmic scaling is used.
        """
        traversal_postorder = _postorder(
            self.node_left_child, self.node_right_sib, self.node_left_child[-1]
        )
        if self.log_scale:
            self.node_variant_likelihood = self._log_likelihood_felsenstein(
                traversal_postorder
            )
            self.affected_node_likelihood = np.full(
                (self.num_nodes, self.call_genotype.shape[0], 4),
                -np.inf,
                dtype=np.float64,
            )
        else:
            self.node_variant_likelihood = self._likelihood_felsenstein(
                traversal_postorder
            )
            self.affected_node_likelihood = np.zeros_like(self.node_variant_likelihood)

    def init_sample_node_mask(self):
        """
        Create a mask for sample nodes.
        """
        sample_node_mask = np.zeros(self.num_nodes, dtype=np.bool_)
        sample_node_mask[self.sample_node] = True
        return sample_node_mask

    def spr(self, source, destination):
        """
        Perform a Subtree Pruning and Regrafting (SPR) operation
        between source and destination nodes.

        :param int source: The node to prune.
        :param int destination: The node to regraft.
        :return: The reverse SPR node.
        """

        reverse_spr = -1

        src_parent = self.node_parent[source]
        dest_parent = self.node_parent[destination]
        src_parent_parent = self.node_parent[src_parent]

        if self.node_left_child[src_parent] == source:
            reverse_spr = self.node_right_sib[source]
        elif self.node_right_sib[self.node_left_child[src_parent]] == source:
            reverse_spr = self.node_left_child[src_parent]

        if self.node_left_child[src_parent_parent] == src_parent:
            if self.node_left_child[src_parent] == source:
                self.node_right_sib[source] = -1
            elif self.node_right_sib[self.node_left_child[src_parent]] == source:
                self.node_right_sib[reverse_spr] = -1

            self.node_left_child[src_parent_parent] = reverse_spr
            self.node_right_sib[reverse_spr] = self.node_right_sib[src_parent]
            self.node_parent[reverse_spr] = src_parent_parent
            self.node_right_sib[src_parent] = -1
        else:
            if self.node_left_child[src_parent] == source:
                self.node_right_sib[source] = -1
            elif self.node_right_sib[self.node_left_child[src_parent]] == source:
                self.node_right_sib[reverse_spr] = -1

            self.node_parent[reverse_spr] = src_parent_parent
            self.node_right_sib[self.node_left_child[src_parent_parent]] = reverse_spr

        self.node_branch_length[reverse_spr] += self.node_branch_length[src_parent]
        self.node_parent[src_parent] = -1
        self.node_branch_length[src_parent] = 0

        if self.node_left_child[dest_parent] == destination:
            self.node_right_sib[src_parent] = self.node_right_sib[destination]

            self.node_left_child[src_parent] = destination
            self.node_parent[destination] = src_parent

            self.node_left_child[dest_parent] = src_parent
            self.node_parent[src_parent] = dest_parent

            self.node_right_sib[destination] = source

        elif self.node_right_sib[self.node_left_child[dest_parent]] == destination:
            self.node_right_sib[self.node_left_child[dest_parent]] = src_parent

            self.node_parent[src_parent] = dest_parent

            self.node_left_child[src_parent] = destination
            self.node_parent[destination] = src_parent
            self.node_right_sib[destination] = source

        dest_branch_length = self.node_branch_length[destination]
        self.node_branch_length[destination] = dest_branch_length / 2
        self.node_branch_length[src_parent] = dest_branch_length / 2

        return reverse_spr

    def subtree_prune(self, source):
        """
        Detaches the subtree rooted at the source node from the tree.

        :param int source: The node to prune.
        :return: The reverse SPR node and the original branch length.
        """
        src_parent = self.node_parent[source]
        src_parent_parent = self.node_parent[src_parent]

        reverse_spr = -1

        # Determine what to do based on the child-sibling relationship
        if self.node_left_child[src_parent] == source:
            reverse_spr = self.node_right_sib[source]
        elif self.node_right_sib[self.node_left_child[src_parent]] == source:
            reverse_spr = self.node_left_child[src_parent]

        if self.node_left_child[src_parent] == source:
            reverse_spr = self.node_right_sib[source]
        elif self.node_right_sib[self.node_left_child[src_parent]] == source:
            reverse_spr = self.node_left_child[src_parent]

        if self.node_left_child[src_parent_parent] == src_parent:
            if self.node_left_child[src_parent] == source:
                self.node_right_sib[source] = -1

            elif self.node_right_sib[self.node_left_child[src_parent]] == source:
                self.node_right_sib[reverse_spr] = -1

            self.node_left_child[src_parent_parent] = reverse_spr
            self.node_right_sib[reverse_spr] = self.node_right_sib[src_parent]
            self.node_parent[reverse_spr] = src_parent_parent
            self.node_right_sib[src_parent] = -1
        else:
            if self.node_left_child[src_parent] == source:
                self.node_right_sib[source] = -1
            elif self.node_right_sib[self.node_left_child[src_parent]] == source:
                self.node_right_sib[reverse_spr] = -1
            self.node_parent[reverse_spr] = src_parent_parent
            self.node_right_sib[self.node_left_child[src_parent_parent]] = reverse_spr

        # Update branch lengths
        reverse_spr_branch_length = self.node_branch_length[src_parent]
        self.node_branch_length[reverse_spr] += self.node_branch_length[src_parent]
        self.node_parent[src_parent] = -1
        self.node_branch_length[src_parent] = 0

        return reverse_spr, reverse_spr_branch_length

    def subtree_regraft(self, source, destination, branch_length=np.inf):
        """
        Reattaches the subtree rooted at the source node to the destination node.

        :param int source: The node to prune.
        :param int destination: The node to regraft.
        :param float branch_length: The branch length between the `src_parent`
        and `dest_parent` nodes.
        """
        src_parent = self.node_parent[source]
        dest_parent = self.node_parent[destination]

        if self.node_left_child[dest_parent] == destination:
            self.node_right_sib[src_parent] = self.node_right_sib[destination]

            self.node_left_child[src_parent] = destination
            self.node_parent[destination] = src_parent

            self.node_left_child[dest_parent] = src_parent
            self.node_parent[src_parent] = dest_parent

            self.node_right_sib[destination] = source

        elif self.node_right_sib[self.node_left_child[dest_parent]] == destination:
            self.node_right_sib[self.node_left_child[dest_parent]] = src_parent

            self.node_parent[src_parent] = dest_parent

            self.node_left_child[src_parent] = destination
            self.node_parent[destination] = src_parent
            self.node_right_sib[destination] = source

        dest_branch_length = self.node_branch_length[destination]

        if branch_length == np.inf:
            branch_length = dest_branch_length / 2
        elif branch_length <= 0:
            raise ValueError("Branch length cannot be 0 or negative.")

        self.node_branch_length[destination] = dest_branch_length - branch_length
        self.node_branch_length[src_parent] = branch_length

    def is_valid_destination(self, source, destination):
        """
        Determines if the source node is on the same path as the destination node.

        :param int source: The source node.
        :param int destination: The destination node.
        :return: True if the source node is not on the same path as the destination node.
        """
        parent = self.node_parent
        u = parent[destination]
        while u != -1:
            if u == source:
                return False
            u = parent[u]
        return True

    def validate_spr(self, source, destination):
        """
        Validates if the Subtree Pruning and Regrafting (SPR) operation
        between source and destination is feasible.

        :param int source: The source node.
        :param int destination: The destination node.
        :return: True if the SPR operation is feasible.
        """
        src_parent = self.node_parent[source]
        src_parent_parent = self.node_parent[src_parent]
        dest_parent = self.node_parent[destination]
        virtual_root = self.num_nodes - 1
        root = self.node_left_child[virtual_root]

        if not self.is_valid_destination(source, destination):
            return False

        if (
            source == destination
            or source < 0
            or destination < 0
            or source >= self.num_nodes
            or destination >= self.num_nodes
            or destination == src_parent
            or src_parent == dest_parent
            or self.locked_nodes[source]
            or src_parent == root
            or src_parent_parent == virtual_root
            or dest_parent == virtual_root
            or destination == virtual_root
            or destination == root
        ):
            return False
        return True

    def _transition_matrix(self, t, mu, pi):
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

    def _calculate_likelihood(self, likelihood, branch_length, rate, pi):
        """
        Calculates likelihoods based on provided child likelihoods,
        branch length, mutation rate, and base frequencies.

        :param np.array likelihood: The likelihoods of the child nodes.
        :param float branch_length: The branch length.
        :param float rate: The mutation rate.
        :param np.array pi: The base frequencies.
        :return: The transition likelihood to the parent node for all variants.
        """
        transition_matrix = self._transition_matrix(branch_length, rate, pi)
        return np.dot(likelihood, transition_matrix)

    def _likelihood_felsenstein(self, traversal_postorder):
        """
        Calculates the likelihood of a tree using Felsenstein's algorithm.

        :param np.array traversal_postorder: The postorder traversal of the tree.
        :return: The likelihood of the tree.
        """
        node_variant_likelihood = np.zeros(
            (self.num_nodes, self.call_genotype.shape[0], 4), dtype=np.float64
        )
        for j, sample_node in enumerate(self.sample_node):
            for i in range(self.call_genotype.shape[0]):
                likelihood = node_variant_likelihood[sample_node]
                if self.call_genotype[i, j] == -1:
                    likelihood[i] = 0.25
                else:
                    likelihood[
                        i, self.variant_allele[i, self.call_genotype[i, j, 0]]
                    ] = 1.0

        for node in traversal_postorder:
            if not self.sample_node_mask[node]:
                u = self.node_left_child[node]
                probs = self._calculate_likelihood(
                    node_variant_likelihood[u],
                    self.node_branch_length[u],
                    self.rate,
                    self.pi,
                )
                v = self.node_right_sib[u]
                while v != -1:
                    probs *= self._calculate_likelihood(
                        node_variant_likelihood[v],
                        self.node_branch_length[v],
                        self.rate,
                        self.pi,
                    )
                    v = self.node_right_sib[v]
                node_variant_likelihood[node] = probs
        return node_variant_likelihood

    def update_temp_likelihoods_upward(self, node):
        """
        Updates the temporary affected node likelihoods upward from a specific node.

        :param int node: The node to update.
        """

        parent = self.node_parent[node]
        while parent != -1:
            self.affected_node[parent] = True
            u = self.node_left_child[parent]
            if self.affected_node[u]:
                likelihood = self.affected_node_likelihood[u]
            else:
                likelihood = self.node_variant_likelihood[u]
            probs = self._calculate_likelihood(
                likelihood,
                self.node_branch_length[u],
                self.rate,
                self.pi,
            )
            v = self.node_right_sib[u]
            while v != -1:
                if self.affected_node[v]:
                    likelihood = self.affected_node_likelihood[v]
                else:
                    likelihood = self.node_variant_likelihood[v]
                probs *= self._calculate_likelihood(
                    likelihood,
                    self.node_branch_length[v],
                    self.rate,
                    self.pi,
                )
                v = self.node_right_sib[v]

            self.affected_node_likelihood[parent] = probs

            parent = self.node_parent[parent]

    def update_likelihood_upward(self, node):
        """
        Updates the likelihoods of the tree upward from specific node.

        :param int node: The node to update.
        """
        parent = self.node_parent[node]
        while parent != -1:
            u = self.node_left_child[parent]
            likelihood = self.node_variant_likelihood[u]
            probs = self._calculate_likelihood(
                likelihood,
                self.node_branch_length[u],
                self.rate,
                self.pi,
            )
            v = self.node_right_sib[u]
            while v != -1:
                likelihood = self.node_variant_likelihood[v]
                probs *= self._calculate_likelihood(
                    likelihood,
                    self.node_branch_length[v],
                    self.rate,
                    self.pi,
                )
                v = self.node_right_sib[v]

            self.node_variant_likelihood[parent] = probs

            parent = self.node_parent[parent]

    def clean_temp_nodes(self, node):
        """
        Cleans the temporary nodes upward from a specific node.

        :param int node: The node to clean.
        """
        u = self.node_parent[node]
        while u != -1:
            self.affected_node[u] = False
            self.affected_node_likelihood[u] = 0.0
            u = self.node_parent[u]

    def random_hill_climbing(self, current_max, num_moves=10):
        """
        Performs a series of random SPR operations to explore tree configurations
        that might increase a likelihood score.

        :param float current_max: The current maximum likelihood.
        :param int num_moves: The number of random moves to make.
        :return max_value: The maximum likelihood.
        :return best_source: The best source node.
        :return best_destination: The best destination node.
        """
        max_value = current_max
        best_source, best_destination = -1, -1

        for _ in range(num_moves):
            source, destination = np.random.randint(0, self.num_nodes, 2)
            if self.validate_spr(source, destination):
                reverse_spr, reverse_spr_branch_length = self.subtree_prune(source)
                self.update_temp_likelihoods_upward(reverse_spr)
                self.subtree_regraft(source, destination)
                self.update_temp_likelihoods_upward(source)

                root_likelihood = self.affected_node_likelihood[
                    self.node_left_child[-1]
                ]
                value = np.prod(np.sum(root_likelihood * 0.25, axis=1))

                self.clean_temp_nodes(source)
                self.clean_temp_nodes(reverse_spr)

                if value > max_value:
                    max_value = value
                    best_source, best_destination = source, destination

                self.subtree_prune(source)
                self.subtree_regraft(source, reverse_spr, reverse_spr_branch_length)

        return max_value, best_source, best_destination

    def _transition_matrix_log(self, t, mu, pi):
        """
        Calculates the logarithmic transition matrix for a given branch length,
        mutation rate, and base frequencies.

        :param float t: The branch length.
        :param float mu: The mutation rate.
        :param np.array pi: The base frequencies.
        :return: The logarithmic transition matrix.
        """
        exp_term = np.exp(-mu * t)

        transition_matrix = (1 - exp_term) * pi

        np.fill_diagonal(transition_matrix, exp_term + (1 - exp_term) * pi)

        return np.log(transition_matrix)

    def manual_max(self, arr):
        """
        Calculates the maximum value of an array manually.

        :param np.array arr: The array to calculate the maximum value.
        :return: The maximum value.
        """
        result = np.empty((arr.shape[0],), dtype=arr.dtype)
        for i in range(arr.shape[0]):
            max_value = arr[i, 0]
            for j in range(1, arr.shape[1]):
                if arr[i, j] > max_value:
                    max_value = arr[i, j]
            result[i] = max_value
        return result

    def _log_calculate_likelihood(self, log_likelihood, branch_length, rate, pi):
        """
        Calculates the likelihoods based on provided child likelihoods,
        branch length, mutation rate, and base frequencies.

        :param np.array log_likelihood: The logarithmic likelihoods of the child nodes.
        :param float branch_length: The branch length.
        :param float rate: The mutation rate.
        :param np.array pi: The base frequencies.
        :return: The logarithmic transition likelihood to the
        parent node for all variants.
        """

        log_transition_matrix = self._transition_matrix_log(branch_length, rate, pi)

        # Use log-sum-exp to combine the log-likelihoods
        # with the log-transition probabilities
        max_log = self.manual_max(log_likelihood).reshape(-1, 1)
        exp_term = np.exp(log_likelihood - max_log)
        new_log_likelihood = max_log + np.log(
            np.dot(exp_term, np.exp(log_transition_matrix))
        )

        return new_log_likelihood

    def update_temp_log_likelihoods_upward(self, node):
        """
        Updates the temporary affected node log likelihoods upward from a specific node.

        :param int node: The node to update.
        """
        parent = self.node_parent[node]
        while parent != -1:
            self.affected_node[parent] = True
            u = self.node_left_child[parent]
            if self.affected_node[u]:
                likelihood = self.affected_node_likelihood[u]
            else:
                likelihood = self.node_variant_likelihood[u]
            probs = self._log_calculate_likelihood(
                likelihood,
                self.node_branch_length[u],
                self.rate,
                self.pi,
            )
            v = self.node_right_sib[u]
            while v != -1:
                if self.affected_node[v]:
                    likelihood = self.affected_node_likelihood[v]
                else:
                    likelihood = self.node_variant_likelihood[v]
                probs += self._log_calculate_likelihood(
                    likelihood,
                    self.node_branch_length[v],
                    self.rate,
                    self.pi,
                )
                v = self.node_right_sib[v]

            self.affected_node_likelihood[parent] = probs

            parent = self.node_parent[parent]

    def clean_temp_log_likelihoods_upward(self, node):
        """
        Cleans the temporary log likelihoods upward from a specific node.

        :param int node: The node to clean.
        """
        u = self.node_parent[node]
        while u != -1:
            self.affected_node[u] = False
            self.affected_node_likelihood[u] = -np.inf
            u = self.node_parent[u]

    def update_log_likelihood_upward(self, node):
        """
        Updates the log likelihoods of the tree upward from specific node.

        :param int node: The node to update.
        """
        parent = self.node_parent[node]
        while parent != -1:
            u = self.node_left_child[parent]
            likelihood = self.node_variant_likelihood[u]
            probs = self._log_calculate_likelihood(
                likelihood,
                self.node_branch_length[u],
                self.rate,
                self.pi,
            )
            v = self.node_right_sib[u]
            while v != -1:
                likelihood = self.node_variant_likelihood[v]
                probs += self._log_calculate_likelihood(
                    likelihood,
                    self.node_branch_length[v],
                    self.rate,
                    self.pi,
                )
                v = self.node_right_sib[v]

            self.node_variant_likelihood[parent] = probs

            parent = self.node_parent[parent]

    def log_random_hill_climbing(self, current_max, num_moves=10):
        """
        Performs a series of random SPR operations to explore tree configurations
        that might increase a likelihood score.

        :param float current_max: The current maximum likelihood.
        :param int num_moves: The number of random moves to make.
        :return max_value: The maximum likelihood in logarithmic scale.
        :return best_source: The best source node.
        :return best_destination: The best destination node.
        """
        max_value = current_max
        value = -1
        best_source, best_destination = -1, -1
        for _ in range(num_moves):
            source, destination = np.random.randint(0, self.num_nodes, 2)
            if self.validate_spr(source, destination):
                reverse_spr, reverse_spr_branch_length = self.subtree_prune(source)

                self.update_temp_log_likelihoods_upward(reverse_spr)

                self.subtree_regraft(source, destination)

                self.update_temp_log_likelihoods_upward(source)

                root_likelihood = self.affected_node_likelihood[
                    self.node_left_child[-1]
                ]
                value = np.sum(np.log(np.sum(np.exp(root_likelihood) * 0.25, axis=1)))

                self.clean_temp_log_likelihoods_upward(source)
                self.clean_temp_log_likelihoods_upward(reverse_spr)

                if value > max_value:
                    max_value = value
                    best_source, best_destination = source, destination

                self.subtree_prune(source)
                self.subtree_regraft(source, reverse_spr, reverse_spr_branch_length)
        return max_value, best_source, best_destination

    def _log_likelihood_felsenstein(self, traversal_postorder):
        """
        Calculates the log likelihood of a tree using
        Felsenstein's algorithm in logarithmic scale.

        :param np.array traversal_postorder: The postorder traversal of the tree.
        :return: The log likelihood of the tree.
        """
        num_nodes = self.num_nodes
        genotypes = self.call_genotype
        num_variants = genotypes.shape[0]
        log_0_25 = np.log(0.25)

        node_variant_likelihood = np.full(
            (num_nodes, num_variants, 4), -np.inf, dtype=np.float64
        )

        for j, sample_node in enumerate(self.sample_node):
            for i in range(num_variants):
                likelihood = node_variant_likelihood[sample_node]
                if self.call_genotype[i, j] == -1:
                    likelihood[i] = log_0_25
                else:
                    likelihood[
                        i, self.variant_allele[i, self.call_genotype[i, j, 0]]
                    ] = 0.0  # log(1) = 0

        for node in traversal_postorder:
            if self.sample_node_mask[node]:
                continue

            u = self.node_left_child[node]
            node_likelihood = self._log_calculate_likelihood(
                node_variant_likelihood[u],
                self.node_branch_length[u],
                self.rate,
                self.pi,
            )

            v = self.node_right_sib[u]
            while v != -1:
                node_likelihood += self._log_calculate_likelihood(
                    node_variant_likelihood[v],
                    self.node_branch_length[v],
                    self.rate,
                    self.pi,
                )
                v = self.node_right_sib[v]

            node_variant_likelihood[node] = node_likelihood

        return node_variant_likelihood
