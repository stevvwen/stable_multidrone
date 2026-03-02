import numpy as np
import scipy.linalg
import math

R_EARTH = 6378137.0  # Earth radius

def ll_to_en(lat, lon, lat0, lon0):
    """Convert lat/lon to local East-North (meters) around reference (lat0, lon0)."""
    
    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    x = R_EARTH * dlon * math.cos(math.radians(lat0))  # East
    y = R_EARTH * dlat                                # North
    return x, y

def en_to_ll(east, north, lat0, lon0):
    """Inverse of ll_to_en: (E,N) meters -> (lat, lon) degrees around (lat0,lon0)."""
    lat = lat0 + math.degrees(north / R_EARTH)
    lon = lon0 + math.degrees(east / (R_EARTH * math.cos(math.radians(lat0))))
    return lat, lon

def ll_to_en_batch(ll, lat0, lon0):
    """Vectorized (n,2) [lat,lon] -> (n,2) [east,north] in local meters around (lat0, lon0).

    Uses the midpoint latitude for the cos term (slightly more accurate over larger spans).
    """
    lat_r = np.radians(ll[:, 0])
    lon_r = np.radians(ll[:, 1])
    lat0_r = math.radians(lat0)
    lon0_r = math.radians(lon0)
    x = R_EARTH * (lon_r - lon0_r) * np.cos((lat_r + lat0_r) * 0.5)
    y = R_EARTH * (lat_r - lat0_r)
    return np.column_stack([x, y])

def en_to_ll_batch(xy, lat0, lon0):
    """Vectorized (n,2) [east,north] -> (n,2) [lat,lon] around (lat0, lon0)."""
    lat0_r = math.radians(lat0)
    lon0_r = math.radians(lon0)
    lat_r = lat0_r + xy[:, 1] / R_EARTH
    lon_r = lon0_r + xy[:, 0] / (R_EARTH * np.cos(lat_r))
    return np.column_stack([np.degrees(lat_r), np.degrees(lon_r)])

class KalmanFilterLatLonCV:
    """
    Constant-velocity Kalman filter in local EN frame (meters).

    State (4D): [E, N, vE, vN]   (m, m, m/s, m/s)
    Measurement (2D): [lat, lon] (deg)  <-- will be converted to (E,N) internally

    - dt in seconds
    - q_acc: white-acceleration power spectral density (m^2/s^3)
    - r_meas: position stdev in meters (assumed isotropic)
    """

    def __init__(self, lat0, lon0, dt=0.1, q_acc=0.1, r_meas=3.0, p_pos0= 25, p_vel0= 100):
        self.lat0 = float(lat0)
        self.lon0 = float(lon0)

        ndim, dt = 2, float(dt)
        self._ndim = ndim
        self._dt = dt

        # State transition for CV
        self.F = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self.F[i, ndim + i] = dt

        # Observation: we observe position only (E,N)
        self.H = np.eye(ndim, 2 * ndim)  # picks out [E, N]

        # Process noise (discrete IWPA)
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2
        q = float(q_acc)
        block = np.array([[dt4/4.0, dt3/2.0],
                          [dt3/2.0, dt2     ]], dtype=float) * q
        self._Q = np.zeros((4, 4), dtype=float)
        self._Q[0:2, 0:2] = block[0,0] * np.eye(2)
        self._Q[0:2, 2:4] = block[0,1] * np.eye(2)
        self._Q[2:4, 0:2] = block[1,0] * np.eye(2)
        self._Q[2:4, 2:4] = block[1,1] * np.eye(2)

        # Measurement noise in meters
        r = float(r_meas)
        self._R = (r * r) * np.eye(ndim)

        # Initial covariance
        self._P0 = np.diag([p_pos0, p_pos0, p_vel0, p_vel0]).astype(float)

    # ---------- Public API (same names as your class) ----------

    def initiate(self, measurement_ll):
        """
        Create track from first lat/lon measurement.

        measurement_ll : (2,) array-like [lat, lon] in degrees
        Returns: mean (4,), covariance (4,4)
        """
        lat, lon = float(measurement_ll[0]), float(measurement_ll[1])
        E, N = ll_to_en(lat, lon, self.lat0, self.lon0)
        mean = np.array([E, N, 0.0, 0.0], dtype=float)
        covariance = self._P0.copy()
        return mean, covariance

    def predict(self, mean, covariance):
        mean = self.F @ mean
        covariance = self.F @ covariance @ self.F.T + self._Q
        return mean, covariance

    def project(self, mean, covariance):
        """
        Project to measurement space (E,N) — note: still EN, not lat/lon.
        We convert the *incoming* measurements to EN, so comparing in EN is consistent.
        """
        z_mean = self.H @ mean                      # (E_pred, N_pred)
        S = self.H @ covariance @ self.H.T + self._R
        return z_mean, S

    def multi_predict(self, mean, covariance):
        mean_pred = mean @ self.F.T
        left = (self.F @ covariance).transpose((1, 0, 2))
        cov_pred = (left @ self.F.T) + self._Q
        return mean_pred, cov_pred

    def update(self, x, P, z_ll):
        """
        z_ll: measurement [lat, lon] in degrees — converted to (E,N) internally.
        """
        zE, zN = ll_to_en(float(z_ll[0]), float(z_ll[1]), self.lat0, self.lon0)
        z = np.array([zE, zN], dtype=float)

        z_mean, S = self.project(x, P)
        innovation = z - z_mean

        K = P @ self.H.T @ np.linalg.inv(S)
        newx = x + K @ innovation

        I = np.eye(self.F.shape[0])
        P = (I - K @ self.H) @ P @ (I - K @ self.H).T + K @ self._R @ K.T  # Joseph form
        newP = 0.5 * (P + P.T)
        return newx, newP

    def gating_distance(self, mean, covariance, measurements_ll, metric='maha'):
        """
        measurements_ll : (N,2) array of [lat, lon] measurements (deg).
        Converts to EN and computes distance in EN.
        """
        # Convert all to EN
        EN = np.array([ll_to_en(lat, lon, self.lat0, self.lon0) for lat, lon in measurements_ll], dtype=float)
        z_mean, S = self.project(mean, covariance)
        d = EN - z_mean

        if metric == 'gaussian':
            return np.sum(d * d, axis=1)
        elif metric == 'maha':
            factor = np.linalg.cholesky(S)
            z = scipy.linalg.solve_triangular(factor, d.T, lower=True, check_finite=False, overwrite_b=True)
            return np.sum(z * z, axis=0)
        else:
            raise ValueError('invalid distance metric')

    # ---------- Convenience helpers ----------

    def state_ll(self, mean):
        """Return (lat, lon) for the current position part of the state."""
        E, N = float(mean[0]), float(mean[1])
        return en_to_ll(E, N, self.lat0, self.lon0)

    def meas_en(self, measurement_ll):
        """Utility to see the EN measurement you fed in."""
        return ll_to_en(float(measurement_ll[0]), float(measurement_ll[1]), self.lat0, self.lon0)

def haversine(lat1, lon1, lat2, lon2):
    # Convert degrees to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Compute deltas
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


    # Calculate and return distance
    return R_EARTH * c



def positional_estimate(drone_lat, drone_lon, altitude, drone_angle, 
                        drone_heading, camera_angle, pixel_x, pixel_y):
    """
    Estimate the GPS coordinates of an object in an image taken from a drone.
    Parameters:
    - drone_lat (float): Latitude of the drone in degrees.
    - drone_lon (float): Longitude of the drone in degrees.
    - altitude (float): Altitude of the drone in feet.
    - drone_angle (float): Angle of the drone in degrees.
    - drone_heading (float): Heading of the drone in degrees (0° is North,
                                90° is East, 180° is South, 270° is West).
    - camera_angle (float): Angle of the camera in degrees.
    - pixel_x (int): X coordinate of the object in the image (in pixels).
    - pixel_y (int): Y coordinate of the object in the image (in pixels).
    
    """

    # Image and camera parameters
    image_width = 3840
    image_height = 2160
    focal_length=  24
    sensor_width = 36.0

    # Earth's approximate conversions
    feet_per_degree_lat = 364000  # Approximate feet per degree of latitude


    heading_rad = np.radians(drone_heading )
    camera_angle = 90 + camera_angle + drone_angle

    feet_per_degree_lon = feet_per_degree_lat * np.cos(np.radians(drone_lat))  # Adjusted for latitude

    sensor_height = sensor_width * (image_height / image_width)  # Maintain aspect ratio

    # Compute angular displacement from center
    delta_x = pixel_x - (image_width / 2)  # Left/right shift
    delta_y = pixel_y - (image_height / 2)  # Up/down shift


    theta_x = np.arctan(delta_x*sensor_width/(image_width*focal_length))# Horizontal angle shift
    theta_y = np.arctan(delta_y*sensor_height/(image_height*focal_length))# Vertical angle shift


    # Compute new ground distance
    D = altitude * np.tan(np.radians(camera_angle) + theta_y)

    # Compute lateral displacement due to theta_x
    Dp= np.sqrt(D**2 + altitude**2)     
    Dx= Dp* np.tan(theta_x) # Lateral distance
    Dr= np.sqrt((np.sin(theta_x)*altitude)**2 + D**2 )/np.cos(theta_x) # New ground distance
    beta= np.arcsin(Dx/Dr) if Dr!= 0 else 0 # Lateral angle shift

    # Compute new offsets including lateral shift
    delta_north_object = Dr * np.cos(heading_rad+ beta)
    delta_east_object = Dr * np.sin(heading_rad+ beta)
  
    # Convert ground displacement from feet to degrees and Calculate final GPS coordinates.
    object_lat_final = drone_lat + delta_north_object / feet_per_degree_lat
    object_lon_final = drone_lon + delta_east_object / feet_per_degree_lon

    return object_lat_final, object_lon_final