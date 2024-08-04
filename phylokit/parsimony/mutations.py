def fitch_algorithm(postorder, left_child, right_sib):
    """
    This is a naive implementation of encoding a node's partition
    using python's set operations to calculate the unweighted robinson
    foulds distance (symmetric distance)
    NOTE: This pure python implementation is faster than the jitted implementation,
    the speed of traversing the tree is faster in jitted version, but it took to
    much time calculating bitwise_or, therefore we use the naive implementation for
    now.
    NOTE: A potential optimization is using bit shifting rather than bitwise_or
    """
    tree_sets = [set()] * (len(left_child) - 1)
    for u in postorder:
        v = left_child[u]
        if v == -1:
            tree_sets[u] = {u}
        else:
            while v != -1:
                tree_sets[u] = (
                    tree_sets[u].intersection(tree_sets[v])
                    if len(tree_sets[u].intersection(tree_sets[v])) == 0
                    else tree_sets[u].union(tree_sets[v])
                )
                v = right_sib[v]

    return tree_sets
