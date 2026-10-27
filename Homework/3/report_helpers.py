"""Report functions for checking the images that actually reached a panorama."""
import itertools
from copy import copy


def audit_rendered_panorama(pipeline, pairs, rendered_paths, max_overlap=.8, min_images=4):
    """Audit rendered inputs only; discarded candidates do not count toward compliance."""
    pairs = list(pairs)
    if not pairs:
        return {'complete': False, 'image_count': 0, 'overlaps': []}
    indices = [pipeline.reader.image_paths.index(path) for path in rendered_paths]
    features = {pipeline.selected_indices[0]: pairs[0][1]}
    features.update({index: second for index, _, second, _ in pairs})
    cached = {(pipeline.selected_indices[i], pair[0]): pair[3]
              for i, pair in enumerate(pairs)}
    audit_pairs = []
    for i, j in itertools.pairwise(indices):
        first, second = features[i], features[j]
        matches = cached.get((i, j))
        if matches is None:
            matches = pipeline.matcher.match(first.keypoints, first.descriptors,
                                             second.keypoints, second.descriptors,
                                             first.image_size, second.image_size)
        audit_pairs.append((j, first, second, matches))
    # The audit uses a separate view so selected/cached results remain intact.
    auditor = copy(pipeline)
    auditor.selected_indices = indices
    rows = auditor.selected_overlap_report(audit_pairs)
    complete = (len(indices) >= min_images and bool(rows)
                and all(row['overlap'] is not None and row['overlap'] <= max_overlap
                        for row in rows))
    return {'complete': complete, 'image_count': len(indices), 'overlaps': rows}
