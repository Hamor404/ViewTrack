import cv2
import matplotlib.pyplot as plt
import numpy as np
from collections import deque

from tracker import matching
from tracker.gmc import GMC
from tracker.basetrack import BaseTrack, TrackState
from tracker.kalman_filter import KalmanFilter

from fast_reid.fast_reid_interfece import FastReIDInterface


class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, feat=None, feat_history=50):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False
        self.depth_vector = self._get_depth_relation_cue()

        self.score = score
        self.tracklet_len = 0

        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)
        self.alpha = 0.9

    def update_features(self, feat):
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        self.features.append(feat)
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def _get_depth_relation_cue(self):
        cx = self._tlwh[0] + 0.5 * self._tlwh[2]
        y2 = self._tlwh[1] + self._tlwh[3]
        cy = self._tlwh[1] + 0.5 * self._tlwh[3]
        x2 = self._tlwh[0]
        if x2 <= 0:
            x2 = 1
        pseudo_depth = 2000 - y2
        bbox_area = self._tlwh[2] * self._tlwh[3]
        return np.asarray([x2, bbox_area, pseudo_depth], dtype=np.float)

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            R = H[:2, :2]
            R8x8 = np.kron(np.eye(4, dtype=float), R)
            t = H[:2, 2]

            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                mean = R8x8.dot(mean)
                mean[:2] += t
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,
                                                               self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @property
    # @jit(nopython=True)
    def depth_relation_cue(self):
        """Convert bounding box to format `((top left, bottom right)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        cx = ret[0] + 0.5 * ret[2]
        x2 = ret[0]
        if x2 <= 0:
            x2 = 1
        y2 = ret[1] + ret[3]
        cy = ret[1] + 0.5 * ret[3]
        pseudo_depth = 2000 - y2
        bbox_area = ret[2] * ret[3]
        return np.asarray([x2, bbox_area, pseudo_depth], dtype=np.float)

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class ViewTracker(object):
    def __init__(self, args, frame_rate=30):

        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []  # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]

        self.frame_id = 0
        self.args = args

        self.track_high_thresh = args.track_high_thresh
        self.track_low_thresh = args.track_low_thresh
        self.new_track_thresh = args.new_track_thresh

        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()
        self.area_proportion = args.area_proportion
        # ReID module
        self.proximity_thresh = args.proximity_thresh
        self.appearance_thresh = args.appearance_thresh

        if args.with_reid:
            self.encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device)

        self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])

    def get_view_type(self, obj):
        area = []
        depth = []
        for t in obj:
            a = (t.depth_relation_cue)[1]
            area.append(a)
            d = (t.depth_relation_cue)[2]
            depth.append(d)
        std_area = np.std(area)
        std_depth = np.std(depth)
        if np.sqrt(std_area) >= std_depth:
            view_type = 'front'
        else:
            view_type = 'top'
        return view_type

    def VAPM(self, obj, level, proportion, viewtype):
        area = []
        depth = []
        for t in obj:
            a = (t.depth_relation_cue)[1]
            area.append(a)
            d = (t.depth_relation_cue)[2]
            depth.append(d)
        # Depth Proportion Partition for front-view
        if viewtype == 'front':
            max_len, mix_len = max(area), min(area)
            if max_len != mix_len:
                r = []
                t = max_len
                while t > mix_len:
                    r.insert(0, t)
                    t = t * proportion
                deep_range = np.array(r)
                if deep_range[0] > mix_len:
                    deep_range = np.concatenate([np.array([mix_len], ), deep_range])
                    deep_range[0] = np.floor(deep_range[0])
                    deep_range[-1] = np.ceil(deep_range[-1])
            else:
                deep_range = [mix_len, ]
            mask = self.get_sub_mask(deep_range, area)
            # if mode == 1:
            #     mask = mask[::-1]

        #  Depth Average Partition for top-view
        else:
            max_len, mix_len = max(depth), min(depth)
            if max_len != mix_len:
                deep_range = np.arange(mix_len, max_len, (max_len - mix_len + 1) / level)
                if deep_range[-1] < max_len:
                    deep_range = np.concatenate([deep_range, np.array([max_len], )])
                    deep_range[0] = np.floor(deep_range[0])
                    deep_range[-1] = np.ceil(deep_range[-1])
            else:
                deep_range = [mix_len, ]
            mask = self.get_sub_mask(deep_range, depth)
        return mask

    def get_deep_range_average(self, obj, step, mode):
        col = []
        for t in obj:
            lend = (t.depth_relation_cue)[mode]
            col.append(lend)
        max_len, mix_len = max(col), min(col)
        if max_len != mix_len:
            deep_range = np.arange(mix_len, max_len, (max_len - mix_len + 1) / step)
            if deep_range[-1] < max_len:
                deep_range = np.concatenate([deep_range, np.array([max_len], )])
                deep_range[0] = np.floor(deep_range[0])
                deep_range[-1] = np.ceil(deep_range[-1])
        else:
            deep_range = [mix_len, ]
        mask = self.get_sub_mask(deep_range, col)
        return mask

    def get_deep_range_proportion(self, obj, proportion, mode):
        col = []
        for t in obj:
            lend = (t.area_vec)[mode]
            col.append(lend)
        max_len, mix_len = max(col), min(col)
        if max_len != mix_len:
            r = []
            t = max_len
            while t > mix_len:
                r.insert(0, t)
                t = t * proportion
            area_range = np.array(r)
            if area_range[0] > mix_len:
                area_range = np.concatenate([np.array([mix_len], ), area_range])
                area_range[0] = np.floor(area_range[0])
                area_range[-1] = np.ceil(area_range[-1])
        else:
            area_range = [mix_len, ]
        mask = self.get_sub_mask(area_range, col)
        # if mode == 1:
        #     mask = mask[::-1]
        return mask

    def get_sub_mask(self, area_range, col):
        mix_len = area_range[0]
        max_len = area_range[-1]
        if max_len == mix_len:
            lc = mix_len
        mask = []
        for d in area_range:
            if d > area_range[0] and d < area_range[-1]:
                mask.append((col >= lc) & (col < d))
                lc = d
            elif d == area_range[-1]:
                mask.append((col >= lc) & (col <= d))
                lc = d
            else:
                lc = d
                continue
        return mask

    def DCM(self, detections, tracks, activated_starcks, refind_stracks, proportion, level, thresh, is_fuse, is_high,
            view_type):
        if len(detections) > 0:
            det_mask = self.VAPM(detections, level, proportion, view_type)
        else:
            det_mask = []

        if len(tracks) > 0:
            track_mask = self.VAPM(tracks, level, proportion, view_type)
        else:
            track_mask = []

        u_detection, u_tracks, res_det, res_track = [], [], [], []

        if len(track_mask) != 0:
            if len(track_mask) < len(det_mask):
                for i in range(len(det_mask) - len(track_mask)):
                    idx = np.argwhere(det_mask[len(track_mask) + i] == True)
                    for idd in idx:
                        res_det.append(detections[idd[0]])

            elif len(track_mask) > len(det_mask):
                for i in range(len(track_mask) - len(det_mask)):
                    idx = np.argwhere(track_mask[len(det_mask) + i] == True)
                    for idd in idx:
                        res_track.append(tracks[idd[0]])

            for dm, tm in zip(det_mask, track_mask):
                det_idx = np.argwhere(dm == True)
                trk_idx = np.argwhere(tm == True)

                # search det
                det_ = []
                for idd in det_idx:
                    det_.append(detections[idd[0]])
                det_ = det_ + u_detection
                # search trk
                track_ = []
                for idt in trk_idx:
                    track_.append(tracks[idt[0]])
                # update trk
                track_ = track_ + u_tracks

                if len(det_) > 0:
                    if not is_high and view_type == 'front':
                        det__mask = self.get_deep_range_proportion(det_, 0.55, 0)
                    else:
                        det__mask = self.get_deep_range_proportion(det_, 0, 0)
                else:
                    det__mask = []
                if len(track_) > 0:
                    if not is_high and view_type == 'front':
                        track__mask = self.get_deep_range_proportion(track_, 0.55, 0)
                    else:
                        track__mask = self.get_deep_range_proportion(track_, 0, 0)
                else:
                    track__mask = []
                u__detection, u__tracks, res__det, res__track = [], [], [], []
                if len(track__mask) != 0:
                    if len(track__mask) < len(det__mask):
                        for i in range(len(det__mask) - len(track__mask)):
                            idx = np.argwhere(det__mask[len(track__mask) + i] == True)
                            for idd in idx:
                                res__det.append(det_[idd[0]])

                    elif len(track__mask) > len(det__mask):
                        for i in range(len(track__mask) - len(det__mask)):
                            idx = np.argwhere(track__mask[len(det__mask) + i] == True)
                            for idd in idx:
                                res__track.append(track_[idd[0]])

                    for dm, tm in zip(det__mask, track__mask):
                        det__idx = np.argwhere(dm == True)
                        trk__idx = np.argwhere(tm == True)

                        # search det
                        det__ = []
                        for idd in det__idx:
                            det__.append(det_[idd[0]])
                        det__ = det__ + u__detection
                        # search trk
                        track__ = []
                        for idt in trk__idx:
                            track__.append(track_[idt[0]])
                        # update trk
                        track__ = track__ + u__tracks
                        if view_type == 'front':
                            ious_dists = matching.iou_distance(track__, det__)
                        else:
                            ious_dists = matching.displacement_iou_distance(track__, det__)
                        ious_dists_mask = (ious_dists > self.proximity_thresh)
                        if (not self.args.mot20) and is_fuse:
                            ious_dists = matching.fuse_score(ious_dists, det__)
                        if self.args.with_reid and is_high:
                            emb_dists = matching.embedding_distance(track__, det__) / 2.0
                            raw_emb_dists = emb_dists.copy()
                            emb_dists_mask = emb_dists > self.appearance_thresh
                            emb_dists[emb_dists_mask] = 1.0
                            emb_dists[ious_dists_mask] = 1.0
                            dists = np.minimum(ious_dists, emb_dists)
                        else:
                            dists = ious_dists

                        matches, u_track_, u_det_ = matching.linear_assignment(dists, thresh)
                        for itracked, idet in matches:
                            track = track__[itracked]
                            det = det__[idet]
                            if track.state == TrackState.Tracked:
                                track.update(det__[idet], self.frame_id)
                                activated_starcks.append(track)
                            else:
                                track.re_activate(det, self.frame_id, new_id=False)
                                refind_stracks.append(track)
                        u__tracks = [track__[t] for t in u_track_]
                        u__detection = [det__[t] for t in u_det_]

                    u_tracks = u__tracks + res__track
                    u_detection = u__detection + res__det

                else:
                    u_detection = det_

            u_tracks = u_tracks + res_track
            u_detection = u_detection + res_det

        else:
            u_detection = detections

        return activated_starcks, refind_stracks, u_tracks, u_detection

    def update(self, output_results, img=None):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []

        if len(output_results):
            if output_results.shape[1] == 5:
                scores = output_results[:, 4]
                bboxes = output_results[:, :4]
                classes = output_results[:, -1]
            else:
                scores = output_results[:, 4] * output_results[:, 5]
                bboxes = output_results[:, :4]  # x1y1x2y2
                classes = output_results[:, -1]

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]

            # Find high threshold detections
            remain_inds = scores > self.args.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]

        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            classes_keep = []

        '''Extract embeddings '''
        if self.args.with_reid:
            features_keep = self.encoder.inference(img, dets)

        if len(dets) > 0:
            '''Detections'''
            if self.args.with_reid:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s, f) for
                              (tlbr, s, f) in zip(dets, scores_keep, features_keep)]
            else:
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                              (tlbr, s) in zip(dets, scores_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        warp = self.gmc.apply(img, dets)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes
        # DCM
        view_type = self.get_view_type(detections)
        print(view_type)
        activated_starcks, refind_stracks, u_track, u_detection_high = self.DCM(
            detections,
            strack_pool,
            activated_starcks,
            refind_stracks,
            self.args.area_proportion_high,
            self.args.depth_levels_high,
            self.args.match_thresh,
            is_fuse=True,
            is_high=True,
            view_type=view_type)

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.args.track_high_thresh
            inds_low = scores > self.args.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
        else:
            dets_second = []
            scores_second = []
            classes_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []
        r_tracked_stracks = [t for t in u_track if t.state == TrackState.Tracked]

        # DCM
        activated_starcks, refind_stracks, u_strack, u_detection_sec = self.DCM(
            detections_second,
            r_tracked_stracks,
            activated_starcks,
            refind_stracks,
            self.args.area_proportion_low,
            self.args.depth_levels_low,
            0.2,
            is_fuse=False,
            is_high=False,
            view_type=view_type)
        for track in u_strack:
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [d for d in u_detection_high]
        dists = matching.iou_distance(unconfirmed, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=self.args.confirm_thresh)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        # get scores of lost tracks
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
