import numpy as np

from utils import util
from utils.gps_util import KalmanFilterLatLonCV, haversine


class State:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class GPSTrack:
    count = 0
    shared_kalman = util.KalmanFilterXYAH()
    gps_kalman = KalmanFilterLatLonCV(lat0=46.2566, lon0=-74.1253)

    def __init__(self, tlwh, score, cls, gps_loc):

        # wait activate
        self._tlwh = np.asarray(self.tlbr_to_tlwh(tlwh[:-1]), dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0
        self.cls = cls
        self.gps_loc = gps_loc
        self.gps_mean, self.gps_covariance = None, None
        self.idx = tlwh[-1]

        # Trajectory storage
        self.trajectory = []

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != State.Tracked:
            mean_state[7] = 0
        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(tracks):
        if len(tracks) <= 0:
            return
        multi_mean = np.asarray([st.mean.copy() for st in tracks])
        multi_covariance = np.asarray([st.covariance for st in tracks])
        for i, st in enumerate(tracks):
            if st.state != State.Tracked:
                multi_mean[i][7] = 0
        multi_mean, multi_covariance = GPSTrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            tracks[i].mean = mean
            tracks[i].covariance = cov

        # Also predict GPS locations
        for st in tracks:
            if st.gps_mean is not None and st.gps_covariance is not None:
                st.gps_mean, st.gps_covariance = GPSTrack.gps_kalman.predict(st.gps_mean, st.gps_covariance)

    def activate(self, kalman_filter, frame_id, object_id=None):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id() if object_id is None else object_id
        self.mean, self.covariance = self.kalman_filter.initiate(self.convert_coords(self._tlwh))

        self.gps_mean, self.gps_covariance = GPSTrack.gps_kalman.initiate(self.gps_loc)

        self.tracklet_len = 0
        self.state = State.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

        self.filtered_gps_loc = GPSTrack.gps_kalman.state_ll(self.gps_mean)

        self.add_trajectory_point(frame_id)

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,
                                                               self.convert_coords(new_track.tlwh))

        self.gps_mean, self.gps_covariance = GPSTrack.gps_kalman.update(self.gps_mean, self.gps_covariance, new_track.gps_loc)
        self.tracklet_len = 0
        self.state = State.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            self.track_id = self.next_id()
        self.score = new_track.score
        self.cls = new_track.cls
        self.idx = new_track.idx
        self.gps_loc = new_track.gps_loc
        self.filtered_gps_loc = GPSTrack.gps_kalman.state_ll(self.gps_mean)

        self.add_trajectory_point(frame_id)

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: Track
        :type frame_id: int
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance,
                                                               self.convert_coords(new_tlwh))

        self.gps_mean, self.gps_covariance = GPSTrack.gps_kalman.update(self.gps_mean, self.gps_covariance, new_track.gps_loc)
        self.state = State.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.cls = new_track.cls
        self.idx = new_track.idx
        self.gps_loc = new_track.gps_loc

        self.filtered_gps_loc = GPSTrack.gps_kalman.state_ll(self.gps_mean)

        self.add_trajectory_point(frame_id)

    def convert_coords(self, tlwh):
        return self.tlwh_to_xyah(tlwh)

    def add_trajectory_point(self, frame_id):
        """Add current position and GPS location to trajectory"""
        bbox = self.tlbr.tolist()

        gps_serializable = self.gps_loc.tolist() if isinstance(self.gps_loc, np.ndarray) else list(self.gps_loc)

        self.trajectory.append({
            'frame_id': int(frame_id),
            'bbox': [float(x) for x in bbox],
            'gps_location': gps_serializable,
            'score': float(self.score),
            'class': int(self.cls),
            "gps_mean": self.gps_mean.tolist() if self.gps_mean is not None else None,
            "gps_covariance": self.gps_covariance.tolist() if self.gps_covariance is not None else None,
            "filtered_gps_loc": list(self.filtered_gps_loc) if self.filtered_gps_loc is not None else None
        })

    def get_trajectory_dict(self):
        """Return trajectory as a dictionary for easy serialization"""
        return {
            'track_id': int(self.track_id),
            'start_frame': int(self.start_frame),
            'end_frame': int(self.end_frame),
            'class': int(self.cls),
            'trajectory': self.trajectory
        }

    def mark_lost(self):
        self.state = State.Lost

    def mark_removed(self):
        self.state = State.Removed

    @property
    def end_frame(self):
        return self.frame_id

    @staticmethod
    def next_id():
        GPSTrack.count += 1
        return GPSTrack.count

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
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

    @staticmethod
    def reset_id():
        GPSTrack.count = 0

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
        return f'OT_{self.track_id}_({self.start_frame}-{self.end_frame})'


class BYTETracker:
    def __init__(self, num_cameras=3, time_lost=60):
        self.tracked_tracks = [[] for _ in range(num_cameras)]
        self.lost_tracks = [[] for _ in range(num_cameras)]
        self.removed_tracks = [[] for _ in range(num_cameras)]

        self.num_cameras = num_cameras
        self.frame_id = 0
        self.max_time_lost = int(time_lost)
        self.kalman_filter = util.KalmanFilterXYAH()
        self.reset_id()

        self.all_trajectories = [[] for _ in range(num_cameras)]

    def update(self, boxes, scores, object_classes, gps_locations, camera_id=0):
        if camera_id == 0:
            self.frame_id += 1
        activated_tracks = []
        re_find_tracks = []
        lost_tracks = []
        removed_tracks = []

        if boxes.shape[0] == 0:
            return np.zeros((0, 10), dtype=np.float32)

        # add index
        boxes = np.concatenate([boxes, np.arange(len(boxes)).reshape(-1, 1)], axis=-1)

        indices_low = scores > 0.1
        indices_high = scores < 0.5
        indices_remain = scores > 0.5

        indices_second = np.logical_and(indices_low, indices_high)
        boxes_remain = boxes[indices_remain]
        boxes_second = boxes[indices_second]
        scores_keep = scores[indices_remain]
        scores_second = scores[indices_second]
        cls_keep = object_classes[indices_remain]
        cls_second = object_classes[indices_second]
        gps_keep = gps_locations[indices_remain]
        gps_second = gps_locations[indices_second]

        detections = self.init_track(boxes_remain, scores_keep, cls_keep, gps_keep)

        unconfirmed = []
        tracked_stracks = []
        for track in self.tracked_tracks[camera_id]:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        # First association, with high score detection boxes
        track_pool = self.joint_stracks(tracked_stracks, self.lost_tracks[camera_id])
        self.multi_predict(track_pool)

        dists = self.get_dists(track_pool, detections)
        matches, u_track, u_detection = util.linear_assignment(dists, thresh=1)
        for tracked_i, box_i in matches:
            track = track_pool[tracked_i]
            det = detections[box_i]
            if track.state == State.Tracked:
                track.update(det, self.frame_id)
                activated_tracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                re_find_tracks.append(track)

        # Second association, with low score detection boxes
        detections_second = self.init_track(boxes_second, scores_second, cls_second, gps_second)
        r_tracked_tracks = [track_pool[i] for i in u_track if track_pool[i].state == State.Tracked]
        dists = util.iou_distance(r_tracked_tracks, detections_second)
        matches, u_track, u_detection_second = util.linear_assignment(dists, thresh=0.5)
        for tracked_i, box_i in matches:
            track = r_tracked_tracks[tracked_i]
            det = detections_second[box_i]
            if track.state == State.Tracked:
                track.update(det, self.frame_id)
                activated_tracks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                re_find_tracks.append(track)

        for it in u_track:
            track = r_tracked_tracks[it]
            if track.state != State.Lost:
                track.mark_lost()
                lost_tracks.append(track)

        # Deal with unconfirmed tracks
        detections = [detections[i] for i in u_detection]
        dists = self.get_dists(unconfirmed, detections)
        matches, u_unconfirmed, u_detection = util.linear_assignment(dists, thresh=0.7)
        for tracked_i, box_i in matches:
            unconfirmed[tracked_i].update(detections[box_i], self.frame_id)
            activated_tracks.append(unconfirmed[tracked_i])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_tracks.append(track)

        # Init new tracks with cross-camera GPS ID assignment
        gps_locations = [detections[i].gps_loc for i in u_detection]

        if camera_id != 0 and len(gps_locations) > 0:
            ref_tracks = [t for t in self.tracked_tracks[0]]
            taken_ids = {t.track_id for t in self.tracked_tracks[camera_id] if t.is_activated}

            cost_mtx = np.zeros((len(u_detection), len(ref_tracks)), dtype=np.float32) \
                                   if len(ref_tracks) > 0 else np.empty((len(u_detection), 0), dtype=np.float32)

            big = 0.0
            for row, det_idx in enumerate(u_detection):
                u_gps = detections[det_idx].gps_loc
                for col, r_t in enumerate(ref_tracks):
                    if r_t.track_id in taken_ids:
                        cost_mtx[row, col] = 1e6
                        continue
                    d = haversine(u_gps[0], u_gps[1], r_t.gps_loc[0], r_t.gps_loc[1])
                    cost_mtx[row, col] = d
                    if d > big:
                        big = d
            big = 1.0 if big == 0.0 else float(big)

            matches, _, _ = util.linear_assignment(cost_mtx, thresh=big + 1)

        for new_i in u_detection:
            track = detections[new_i]
            if track.score < 0.6:
                continue
            object_id = None
            if camera_id != 0:
                for mi, mj in matches:
                    if new_i == mi:
                        object_id = self.tracked_tracks[0][mj].track_id
                        break

            track.activate(self.kalman_filter, self.frame_id, object_id=object_id)
            activated_tracks.append(track)

        # Update state
        for track in self.lost_tracks[camera_id]:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_tracks.append(track)

        self.tracked_tracks[camera_id] = [t for t in self.tracked_tracks[camera_id] if t.state == State.Tracked]
        self.tracked_tracks[camera_id] = self.joint_stracks(self.tracked_tracks[camera_id], activated_tracks)
        self.tracked_tracks[camera_id] = self.joint_stracks(self.tracked_tracks[camera_id], re_find_tracks)
        self.lost_tracks[camera_id] = self.sub_stracks(self.lost_tracks[camera_id], self.tracked_tracks[camera_id])
        self.lost_tracks[camera_id].extend(lost_tracks)
        self.lost_tracks[camera_id] = self.sub_stracks(self.lost_tracks[camera_id], self.removed_tracks[camera_id])
        self.removed_tracks[camera_id].extend(removed_tracks)
        self.tracked_tracks[camera_id], self.lost_tracks[camera_id] = self.remove_duplicate_stracks(self.tracked_tracks[camera_id], self.lost_tracks[camera_id])
        output = [track.tlbr.tolist() + [track.track_id,
                                         track.score,
                                         track.cls,
                                         track.idx, track.gps_loc[0], track.gps_loc[1]] for track in self.tracked_tracks[camera_id]]
        return np.asarray(output, dtype=np.float32)

    @staticmethod
    def init_track(boxes, scores, cls, gps_loc):
        return [GPSTrack(box, s, c, g) for (box, s, c, g) in zip(boxes, scores, cls, gps_loc)] if len(boxes) else []

    @staticmethod
    def get_dists(tracks, detections):
        iou_cost = util.iou_distance(tracks, detections)
        iou_cost = util.fuse_score(iou_cost, detections)

        if len(tracks) == 0 or len(detections) == 0:
            return iou_cost

        gps_cost = np.ones_like(iou_cost, dtype=np.float32)

        gps_gate_m = 20.0
        gps_norm_m = 20.0
        w_iou, w_gps = 0.7, 0.3

        for ti, trk in enumerate(tracks):
            if getattr(trk, "filtered_gps_loc", None) is None:
                continue
            tlat, tlon = trk.filtered_gps_loc
            for di, det in enumerate(detections):
                dlat, dlon = det.gps_loc
                d_m = haversine(tlat, tlon, dlat, dlon)
                if d_m > gps_gate_m:
                    gps_cost[ti, di] = 1e6
                else:
                    gps_cost[ti, di] = min(d_m / gps_norm_m, 1.0)

        fused = w_iou * iou_cost + w_gps * gps_cost
        return fused

    @staticmethod
    def multi_predict(tracks):
        GPSTrack.multi_predict(tracks)

    @staticmethod
    def reset_id():
        GPSTrack.reset_id()

    @staticmethod
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

    @staticmethod
    def sub_stracks(tlista, tlistb):
        stracks = {t.track_id: t for t in tlista}
        for t in tlistb:
            tid = t.track_id
            if stracks.get(tid, 0):
                del stracks[tid]
        return list(stracks.values())

    @staticmethod
    def remove_duplicate_stracks(stracksa, stracksb):
        pdist = util.iou_distance(stracksa, stracksb)
        pairs = np.where(pdist < 0.15)
        dupa, dupb = [], []
        for p, q in zip(*pairs):
            timep = stracksa[p].frame_id - stracksa[p].start_frame
            timeq = stracksb[q].frame_id - stracksb[q].start_frame
            if timep > timeq:
                dupb.append(q)
            else:
                dupa.append(p)
        resa = [t for i, t in enumerate(stracksa) if i not in dupa]
        resb = [t for i, t in enumerate(stracksb) if i not in dupb]
        return resa, resb

    def save_trajectories_csv(self, output_file="trajectories"):
        """Save trajectories to CSV format"""
        import csv

        for c_id in range(self.num_cameras):
            all_tracks_data = []

            for track in self.tracked_tracks[c_id] + self.lost_tracks[c_id] + self.removed_tracks[c_id]:
                if len(track.trajectory) > 0:
                    all_tracks_data.append(track.get_trajectory_dict())

            with open(f"{output_file}_{c_id}.csv", 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['track_id', 'frame_id', 'x1', 'y1', 'x2', 'y2',
                            'latitude', 'longitude', 'score', 'class', 'gps_mean_lat', 'gps_mean_lon'])
                for track_data in all_tracks_data:
                    track_id = track_data['track_id']
                    for traj_point in track_data['trajectory']:
                        frame_id = traj_point['frame_id']
                        bbox = traj_point['bbox']
                        gps_loc = traj_point['gps_location']
                        score = traj_point['score']
                        cls = traj_point['class']
                        gps_mean = traj_point['filtered_gps_loc']

                        lat = float(gps_loc[0])
                        lon = float(gps_loc[1])

                        writer.writerow([track_id, frame_id,
                                    float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]),
                                    lat, lon, float(score), int(cls),
                                    gps_mean[0], gps_mean[1]])

                print(f"Saved trajectories to {output_file}_{c_id}.csv")
