import numpy as np
import scipy
import lap
import torch
from scipy.spatial.distance import cdist

from cython_bbox import bbox_overlaps as bbox_ious
from tracker import kalman_filter


def merge_matches(m1, m2, shape):
    O, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    mask = M1 * M2
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    matched_cost = cost_matrix[tuple(zip(*indices))]
    matched_mask = (matched_cost <= thresh)

    matches = indices[matched_mask]
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float),
        np.ascontiguousarray(btlbrs, dtype=np.float)
    )

    return ious



def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def s_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr_s for track in atracks]
        btlbrs = [track.tlbr_s for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix


def h_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    hious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if hious.size != 0:
        for i in range(len(atlbrs)):
            for j in range(len(btlbrs)):
                h_max = max(abs(atlbrs[i][1] - btlbrs[j][3]) + 1, abs(atlbrs[i][3] - btlbrs[j][1]) + 1)
                h_min = min(abs(atlbrs[i][1] - btlbrs[j][3]) + 1, abs(atlbrs[i][3] - btlbrs[j][1]) + 1)
                h_iou = h_min / h_max
                hious[i][j] = h_iou
    hm_iou = _ious * hious
    cost_matrix = 1 - hm_iou
    return cost_matrix


def w_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    hious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if hious.size != 0:
        for i in range(len(atlbrs)):
            for j in range(len(btlbrs)):
                h_max = max(atlbrs[i][0], btlbrs[j][0])
                h_min = min(atlbrs[i][0], btlbrs[j][0])
                h_iou = h_min / h_max
                hious[i][j] = h_iou
    hm_iou = _ious * hious
    cost_matrix = 1 - hm_iou
    return cost_matrix


def wh_displacement_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)

    axywhs = [track.tlbr_wh for track in atracks]
    bxywhs = [track.tlbr_wh for track in btracks]
    wh_ious = ious(axywhs, bxywhs)

    hm_iou = _ious * wh_ious
    cost_matrix = 1 - hm_iou
    return cost_matrix


def h_displacement_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)

    axywhs = [track.xywh for track in atracks]
    bxywhs = [track.xywh for track in btracks]
    hious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if hious.size != 0:
        for i in range(len(axywhs)):
            for j in range(len(bxywhs)):
                h_1 = abs(axywhs[i][1] - bxywhs[j][1])
                h_2 = (axywhs[i][3] + bxywhs[j][3]) / 2.0
                h_iou = h_1 / h_2
                if h_iou > 1:
                    h_iou = 1
                hious[i][j] = 1 - h_iou
    hm_iou = _ious * hious
    cost_matrix = 1 - hm_iou
    return cost_matrix


def w_displacement_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)

    axywhs = [track.xywh for track in atracks]
    bxywhs = [track.xywh for track in btracks]
    wious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if wious.size != 0:
        for i in range(len(axywhs)):
            for j in range(len(bxywhs)):
                w_1 = abs(axywhs[i][0] - bxywhs[j][0])
                w_2 = (axywhs[i][2] + bxywhs[j][2]) / 2.0
                w_iou = w_1 / w_2
                if w_iou > 1:
                    w_iou = 1
                wious[i][j] = 1 - w_iou
    wm_iou = _ious * wious
    cost_matrix = 1 - wm_iou
    return cost_matrix


def displacement_iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
            len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)

    axywhs = [track.xywh for track in atracks]
    bxywhs = [track.xywh for track in btracks]
    d_ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float)
    if d_ious.size != 0:
        for i in range(len(axywhs)):
            for j in range(len(bxywhs)):
                h_1 = abs(axywhs[i][1] - bxywhs[j][1])
                h_2 = (axywhs[i][3] + bxywhs[j][3]) / 2.0
                w_1 = abs(axywhs[i][0] - bxywhs[j][0])
                w_2 = (axywhs[i][2] + bxywhs[j][2]) / 2.0

                if w_1 > h_1:
                    d_iou = w_1 / w_2
                else:
                    d_iou = h_1 / h_2

                if d_iou > 1:
                    d_iou = 1
                d_ious[i][j] = 1 - d_iou
    dm_iou = _ious * d_ious
    cost_matrix = 1 - dm_iou
    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """
    :param tracks: list[STrack]
    :param detections: list[BaseTrack]
    :param metric:
    :return: cost_matrix np.ndarray
    """

    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float)

    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))  # / 2.0  # Nomalized features
    return cost_matrix


def angle_diff_distance(tracks, detections, velocities, previous_obs, vdc_weight):
    Y, X = speed_direction_batch(detections, previous_obs)
    inertia_Y, inertia_X = velocities[:, 0], velocities[:, 1]
    inertia_Y = np.repeat(inertia_Y[:, np.newaxis], Y.shape[1], axis=1)
    inertia_X = np.repeat(inertia_X[:, np.newaxis], X.shape[1], axis=1)
    diff_angle_cos = inertia_X * X + inertia_Y * Y
    diff_angle_cos = np.clip(diff_angle_cos, a_min=-1, a_max=1)
    diff_angle = np.arccos(diff_angle_cos)
    diff_angle = (np.pi / 2.0 - np.abs(diff_angle)) / np.pi

    valid_mask = np.ones(previous_obs.shape[0])
    valid_mask[np.where(previous_obs[:, 4] < 0)] = 0

    scores = np.repeat(np.array([det.score for det in detections])[:, np.newaxis], len(tracks), axis=1)
    # iou_matrix = iou_matrix * scores # a trick sometiems works, we don't encourage this
    valid_mask = np.repeat(valid_mask[:, np.newaxis], X.shape[1], axis=1)

    angle_diff_cost = (valid_mask * diff_angle) * vdc_weight
    angle_diff_cost = angle_diff_cost.T
    angle_diff_cost = angle_diff_cost * scores

    return angle_diff_cost.T


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    # measurements = np.asarray([det.to_xyah() for det in detections])
    measurements = np.asarray([det.to_xywh() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    # measurements = np.asarray([det.to_xyah() for det in detections])
    measurements = np.asarray([det.to_xywh() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    reid_sim = 1 - cost_matrix
    iou_dist = iou_distance(tracks, detections)
    iou_sim = 1 - iou_dist
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    # fuse_sim = fuse_sim * (1 + det_scores) / 2
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost


def speed_direction_batch(atracks, btracks):
    # tracks = tracks[..., np.newaxis]
    # CX1, CY1 = (dets[:, 0] + dets[:, 2]) / 2.0, (dets[:, 1] + dets[:, 3]) / 2.0
    # CX2, CY2 = (tracks[:, 0] + tracks[:, 2]) / 2.0, (tracks[:, 1] + tracks[:, 3]) / 2.0
    btracks = btracks[..., np.newaxis]
    atlbrs = np.array([track.tlbr for track in atracks])
    CX1, CY1 = (atlbrs[:, 0] + atlbrs[:, 2]) / 2.0, (atlbrs[:, 1] + atlbrs[:, 3]) / 2.0
    CX2, CY2 = (btracks[:, 0] + btracks[:, 2]) / 2.0, (btracks[:, 1] + btracks[:, 3]) / 2.0
    dx = CX1 - CX2
    dy = CY1 - CY2
    norm = np.sqrt(dx ** 2 + dy ** 2) + 1e-6
    dx = dx / norm
    dy = dy / norm
    return dy, dx  # size: num_track x num_det
