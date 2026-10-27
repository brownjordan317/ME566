"""Search ordered image subsets instead of committing to one greedy anchor."""


def find_sequence(count, measure, target_count, max_overlap):
    """Find a target-length sequence, or the longest under measured constraints.

    Adjacent frames need an accepted alignment. Every retained pair must have a
    known overlap no greater than max_overlap. Measurements are cached by caller.
    Search explores all starts and backtracks; it never relaxes overlap limits.
    """
    best = []

    def visit(path):
        nonlocal best
        if len(path) > len(best):
            best = path.copy()
        if len(best) >= target_count:
            return True
        start = path[-1] + 1 if path else 0
        if len(path) + count - start <= len(best):
            return False
        for candidate in range(start, count):
            if path:
                if not measure(path[-1], candidate)['accepted']:
                    continue
                compatible = True
                for previous in path[:-1]:
                    overlap = measure(previous, candidate).get('overlap')
                    if overlap is None or overlap > max_overlap:
                        compatible = False
                        break
                if not compatible:
                    continue
            if visit(path + [candidate]):
                return True
        return False

    visit([])
    return best
