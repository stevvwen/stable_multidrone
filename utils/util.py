import numpy as np
import scipy
import lap
from typing import Optional, Tuple
from math import cos

from utils.gps_util import R_EARTH, ll_to_en, en_to_ll

DEG2RAD = np.pi / 180.0
RAD2DEG = 180.0 / np.pi



def linear_assignment(cost_matrix, thresh, use_lap=True):
    # Linear assignment implementations with scipy and lap.lapjv
    if cost_matrix.size == 0:
        matches = np.empty((0, 2), dtype=int)
        unmatched_a = tuple(range(cost_matrix.shape[0]))
        unmatched_b = tuple(range(cost_matrix.shape[1]))
        return matches, unmatched_a, unmatched_b

    if use_lap:
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
        matches = [[ix, mx] for ix, mx in enumerate(x) if mx >= 0]
        unmatched_a = np.where(x < 0)[0]
        unmatched_b = np.where(y < 0)[0]
    else:
        # Scipy linear sum assignment is NOT working correctly, DO NOT USE
        y, x = scipy.optimize.linear_sum_assignment(cost_matrix)  # row y, col x
        matches = np.asarray([[i, x] for i, x in enumerate(x) if cost_matrix[i, x] <= thresh])
        unmatched = np.ones(cost_matrix.shape)
        for i, xi in matches:
            unmatched[i, xi] = 0.0
        unmatched_a = np.where(unmatched.all(1))[0]
        unmatched_b = np.where(unmatched.all(0))[0]

    return matches, unmatched_a, unmatched_b


def compute_iou(a_boxes, b_boxes):
    """
    Compute cost based on IoU
    :type a_boxes: list[tlbr] | np.ndarray
    :type b_boxes: list[tlbr] | np.ndarray

    :rtype iou | np.ndarray
    """
    iou = np.zeros((len(a_boxes), len(b_boxes)), dtype=np.float32)
    if iou.size == 0:
        return iou
    a_boxes = np.ascontiguousarray(a_boxes, dtype=np.float32)
    b_boxes = np.ascontiguousarray(b_boxes, dtype=np.float32)
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = a_boxes.T
    b2_x1, b2_y1, b2_x2, b2_y2 = b_boxes.T

    # Intersection area
    inter_area = (np.minimum(b1_x2[:, None], b2_x2) - np.maximum(b1_x1[:, None], b2_x1)).clip(0) * \
                 (np.minimum(b1_y2[:, None], b2_y2) - np.maximum(b1_y1[:, None], b2_y1)).clip(0)

    # box2 area
    box1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)
    return inter_area / (box2_area + box1_area[:, None] - inter_area + 1E-7)


def iou_distance(a_tracks, b_tracks):
    """
    Compute cost based on IoU
    :type a_tracks: list[STrack]
    :type b_tracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(a_tracks) > 0 and isinstance(a_tracks[0], np.ndarray)) \
            or (len(b_tracks) > 0 and isinstance(b_tracks[0], np.ndarray)):
        a_boxes = a_tracks
        b_boxes = b_tracks
    else:
        a_boxes = [track.tlbr for track in a_tracks]
        b_boxes = [track.tlbr for track in b_tracks]
    return 1 - compute_iou(a_boxes, b_boxes)  # cost matrix


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim  # fuse_cost


class KalmanFilterXYAH:
    """
    A Kalman filter for tracking bounding boxes in image space.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.

    Object motion follows a constant velocity model. The bounding box location
    (x, y, a, h) is taken as direct observation of the state space (linear
    observation model).

    """

    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        """
        Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are initialized
            to 0 mean.

        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [2 * self._std_weight_position * measurement[3],
               2 * self._std_weight_position * measurement[3],
               1e-2,
               2 * self._std_weight_position * measurement[3],
               10 * self._std_weight_velocity * measurement[3],
               10 * self._std_weight_velocity * measurement[3],
               1e-5,
               10 * self._std_weight_velocity * measurement[3]]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        """
        Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.

        """
        std_pos = [self._std_weight_position * mean[3],
                   self._std_weight_position * mean[3],
                   1e-2,
                   self._std_weight_position * mean[3]]
        std_vel = [self._std_weight_velocity * mean[3],
                   self._std_weight_velocity * mean[3],
                   1e-5,
                   self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        # mean = np.dot(self._motion_mat, mean)
        mean = np.dot(mean, self._motion_mat.T)
        covariance = np.linalg.multi_dot((self._motion_mat, covariance, self._motion_mat.T)) + motion_cov

        return mean, covariance

    def project(self, mean, covariance):
        """
        Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        (ndarray, ndarray)
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [self._std_weight_position * mean[3],
               self._std_weight_position * mean[3],
               1e-1,
               self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((self._update_mat, covariance, self._update_mat.T))
        return mean, covariance + innovation_cov

    def multi_predict(self, mean, covariance):
        """
        Run Kalman filter prediction step (Vectorized version).
        Parameters
        ----------
        mean : ndarray
            The Nx8 dimensional mean matrix of the object states at the previous
            time step.
        covariance : ndarray
            The Nx8x8 dimensional covariance matrix of the object states at the
            previous time step.
        Returns
        -------
        (ndarray, ndarray)
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [self._std_weight_position * mean[:, 3],
                   self._std_weight_position * mean[:, 3],
                   1e-2 * np.ones_like(mean[:, 3]),
                   self._std_weight_position * mean[:, 3]]
        std_vel = [self._std_weight_velocity * mean[:, 3],
                   self._std_weight_velocity * mean[:, 3],
                   1e-5 * np.ones_like(mean[:, 3]),
                   self._std_weight_velocity * mean[:, 3]]
        sqr = np.square(np.r_[std_pos, std_vel]).T

        motion_cov = [np.diag(sqr[i]) for i in range(len(mean))]
        motion_cov = np.asarray(motion_cov)

        mean = np.dot(mean, self._motion_mat.T)
        left = np.dot(self._motion_mat, covariance).transpose((1, 0, 2))
        covariance = np.dot(left, self._motion_mat.T) + motion_cov

        return mean, covariance

    def update(self, mean, covariance, measurement):
        """
        Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        (ndarray, ndarray)
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(mean, covariance)

        chol_factor, lower = scipy.linalg.cho_factor(projected_cov, lower=True, check_finite=False)
        kalman_gain = scipy.linalg.cho_solve((chol_factor, lower),
                                             np.dot(covariance, self._update_mat.T).T,
                                             check_finite=False).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((kalman_gain, projected_cov, kalman_gain.T))
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements, only_position=False, metric='maha'):
        """
        Compute gating distance between state distribution and measurements.
        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.
        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.
        metric : str
            Distance metric.
        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.
        """
        mean, covariance = self.project(mean, covariance)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        d = measurements - mean
        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            return np.sum(z * z, axis=0)  # square maha
        else:
            raise ValueError('invalid distance metric')




class EKF_CV2D:
    """
    Extended Kalman Filter for 2D constant-velocity motion.
    Internal state: [x_east, y_north, vx, vy] in meters and m/s (local ENU).
    Public I/O (when use_latlon_measurements=True): lat/lon in degrees.
    """

    def __init__(
        self,
        *,
        dt: float = 0.1,
        q_acc: float = 1.0,          # accel noise PSD ~ σ_a^2 [m^2/s^3]
        r_meas: float = 3.0,         # ENU pos std [m] if use_latlon_measurements=False
        p_pos0: float = 100.0,       # initial pos variance [m^2]
        p_vel0: float = 1e6,         # initial vel variance [(m/s)^2]
        use_latlon_measurements: bool = True,
        lat0_deg: Optional[float] = None,
        lon0_deg: Optional[float] = None,
        r_meas_latlon: Tuple[float, float] = (5.0, 5.0),  # (σ_lat[m], σ_lon[m]) for R in lat/lon mode
    ):
        self.dt = float(dt)
        self.q_acc = float(q_acc)

        # Mode & origin
        self.use_latlon = bool(use_latlon_measurements)
        self.lat0_deg = lat0_deg
        self.lon0_deg = lon0_deg
        self._origin_set = (self.lat0_deg is not None and self.lon0_deg is not None)

        # Measurement covariance
        if self.use_latlon:
            # Defer building R until origin known (lon scaling needs cos(lat0))
            self._sigma_lat_m, self._sigma_lon_m = r_meas_latlon
            self.R: Optional[np.ndarray] = None
            if self._origin_set:
                self._build_R_from_meters()
        else:
            self.R = (float(r_meas) ** 2) * np.eye(2)

        # Linear measurement matrix for ENU case
        self.H_lin = np.array([[1, 0, 0, 0],
                               [0, 1, 0, 0]], dtype=float)

        # State, covariance, gain
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.K: Optional[np.ndarray] = None

        self._P0 = np.diag([p_pos0, p_pos0, p_vel0, p_vel0]).astype(float)

        # Precompute motion model for current dt
        self.F, self.Q = self._F_Q(self.dt, self.q_acc)

    # ---------- Helpers: build R, conversions ----------
    def _build_R_from_meters(self):
        lat0_rad = self.lat0_deg * DEG2RAD
        sigma_lat_deg = (self._sigma_lat_m / R_EARTH) * RAD2DEG
        sigma_lon_deg = (self._sigma_lon_m / (R_EARTH * cos(lat0_rad))) * RAD2DEG
        self.R = np.diag([sigma_lat_deg**2, sigma_lon_deg**2])

    def _ll_to_enu(self, lat_deg: float, lon_deg: float) -> Tuple[float, float]:
        return ll_to_en(lat_deg, lon_deg, self.lat0_deg, self.lon0_deg)

    def _enu_to_ll(self, x_e: float, y_n: float) -> Tuple[float, float]:
        return en_to_ll(x_e, y_n, self.lat0_deg, self.lon0_deg)

    @staticmethod
    def _F_Q(dt: float, q_acc: float) -> Tuple[np.ndarray, np.ndarray]:
        dt2 = dt * dt
        dt3 = dt2 * dt
        F = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1,  0],
                      [0, 0, 0,  1]], dtype=float)
        q = q_acc
        Q = q * np.array([
            [dt2*dt2/4, 0,          dt3/2,    0],
            [0,          dt2*dt2/4,  0,        dt3/2],
            [dt3/2,      0,          dt2,      0],
            [0,          dt3/2,      0,        dt2]
        ], dtype=float)
        return F, Q

    # ---------- Process model ----------
    def f(self, x: np.ndarray, dt: float) -> np.ndarray:
        if dt != self.dt:
            self.dt = float(dt)
            self.F, self.Q = self._F_Q(self.dt, self.q_acc)
        return self.F @ x

    def F_jac(self, x: np.ndarray, dt: float) -> np.ndarray:
        if dt != self.dt:
            self.dt = float(dt)
            self.F, self.Q = self._F_Q(self.dt, self.q_acc)
        return self.F

    # ---------- Measurement model (position only) ----------
    def h(self, x: np.ndarray) -> np.ndarray:
        # ENU mode: returns ENU [x, y]
        if not self.use_latlon:
            return x[:2].copy()
        # GPS mode: return [lat, lon] deg for current ENU state
        lat, lon = self._enu_to_ll(float(x[0]), float(x[1]))
        return np.array([lat, lon], dtype=float)

    def H_jac(self, x: np.ndarray) -> np.ndarray:
        if not self.use_latlon:
            return self.H_lin
        # Jacobian of [lat,lon] wrt [x,y] at origin latitude
        lat0_rad = self.lat0_deg * DEG2RAD
        dlat_dy = RAD2DEG / R_EARTH
        dlon_dx = RAD2DEG / (R_EARTH * cos(lat0_rad))
        # rows: [d lat / d x, d lat / d y, d lat / d vx, d lat / d vy]
        #       [d lon / d x, d lon / d y, d lon / d vx, d lon / d vy]
        H = np.array([[0.0,    dlat_dy, 0.0, 0.0],
                      [dlon_dx, 0.0,     0.0, 0.0]], dtype=float)
        return H

    # ---------- Public API ----------
    def predict(self, dt: Optional[float] = None):
        if self.x is None:
            raise RuntimeError("Not initialized. Call step(z) or update(z) once with your first measurement.")
        if dt is None:
            dt = self.dt
        Fk = self.F_jac(self.x, dt)
        self.x = self.f(self.x, dt)
        self.P = Fk @ self.P @ Fk.T + self.Q
        self.K = None

    def update(self, z: Tuple[float, float]):
        """
        Update with a measurement:
          - ENU mode: z = [x_m, y_m] in meters
          - GPS mode: z = [lat_deg, lon_deg] in degrees
        """
        z = np.asarray(z, dtype=float).reshape(2)

        # Auto-set origin & build R on first measurement in GPS mode
        if self.x is None:
            if self.use_latlon:
                if not self._origin_set:
                    # Use first fix as origin; build R
                    self.lat0_deg = float(z[0])
                    self.lon0_deg = float(z[1])
                    self._origin_set = True
                    self._build_R_from_meters()
                    # State at origin
                    self.x = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
                else:
                    # Convert first fix to ENU and seed
                    xe, yn = self._ll_to_enu(float(z[0]), float(z[1]))
                    self.x = np.array([xe, yn, 0.0, 0.0], dtype=float)
            else:
                # ENU mode, first fix in meters
                self.x = np.array([z[0], z[1], 0.0, 0.0], dtype=float)

            self.P = self._P0.copy()
            self.K = None
            return

        # Regular EKF update
        Hk = self.H_jac(self.x)
        y  = z - self.h(self.x)
        S  = Hk @ self.P @ Hk.T + self.R  # 2x2
        # K = P H^T S^{-1}  -> solve S * X = H P ; X^T = K (4x2)
        self.K = np.linalg.solve(S, Hk @ self.P).T

        self.x = self.x + self.K @ y
        I  = np.eye(4)
        KH = self.K @ Hk
        self.P = (I - KH) @ self.P @ (I - KH).T + self.K @ self.R @ self.K.T
        self.P = 0.5 * (self.P + self.P.T)  # enforce symmetry

    def step(self, z: Optional[Tuple[float, float]] = None, dt: Optional[float] = None):
        """
        Convenience step:
          - If first call and z is provided -> initialize.
          - Else: predict (optionally with dt), and update if z is not None.
        Returns:
          - GPS mode: (lat_deg, lon_deg)
          - ENU mode: (x_m, y_m)
        """
        if self.x is None:
            if z is None:
                raise RuntimeError("Need a first measurement to initialize.")
            self.update(z)
        else:
            self.predict(dt)
            if z is not None:
                self.update(z)

        return self.position_latlon() if self.use_latlon else tuple(self.x[:2].tolist())

    # ---------- Convenience getters ----------
    def state(self) -> np.ndarray:
        if self.x is None:
            raise RuntimeError("Filter not initialized yet.")
        return self.x.copy()  # ENU meters and m/s

    def covariance(self) -> np.ndarray:
        if self.P is None:
            raise RuntimeError("Filter not initialized yet.")
        return self.P.copy()

    def kalman_gain(self) -> Optional[np.ndarray]:
        return None if self.K is None else self.K.copy()

    def position_latlon(self) -> Tuple[float, float]:
        if not self.use_latlon:
            raise RuntimeError("Filter is in ENU mode; no lat/lon origin set.")
        if self.x is None:
            raise RuntimeError("Filter not initialized yet.")
        return self._enu_to_ll(float(self.x[0]), float(self.x[1]))

    def predict_gps(self, dt: Optional[float] = None) -> Tuple[float, float]:
        """Predict forward and return (lat_deg, lon_deg) without updating."""
        self.predict(dt)
        return self.position_latlon()

    def velocity_mps(self) -> Tuple[float, float]:
        """Return velocity (vx, vy) in meters/second in ENU frame."""
        if self.x is None:
            raise RuntimeError("Filter not initialized yet.")
        return float(self.x[2]), float(self.x[3])