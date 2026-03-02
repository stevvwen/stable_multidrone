import csv
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.ticker import FuncFormatter, MaxNLocator

from math import radians, degrees, cos
import numpy as np
from scipy.spatial import cKDTree

from utils.gps_util import R_EARTH, ll_to_en_batch, en_to_ll_batch


def parse_csv_to_lists(csv_file):
    """Return list of (lat, lon) tuples from a GPS log CSV."""
    lat_lon_list = []
    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lat_lon_list.append((float(row['lat_decimal']), float(row['lon_decimal'])))
    return lat_lon_list

def csv_to_id_dict(csv_file):
    id_dict = defaultdict(dict)  # <-- each ID maps to a dict instead of a list

    with open(csv_file, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                id_val = float(row['ID'])
                time_val = row['time']
                lat = float(row['Latitude'])
                lon = float(row['Longitude'])
                id_dict[id_val][time_val] = (lat, lon)  # <-- dict with time as key
            except (KeyError, ValueError) as e:
                print(f"Skipping invalid row: {row}, error: {e}")

    return dict(id_dict)


def icp_translation_points_gps(
    P_ll: np.ndarray,      # (n,2) source polyline [lat,lon]
    Q_ll: np.ndarray,      # (m,2) target polyline [lat,lon]
    max_iters: int = 30,
    trim_ratio: float = 0.0,   # 0..0.5: drop worst residuals each iter
    damping: float = 1.0,      # 0<damping<=1: 1 = full step
    tol_mm: float = 0.5        # stop when |Δt| < tol_mm
):
    assert P_ll.shape[1] == 2 and Q_ll.shape[1] == 2

    # 1) Fixed ENU origin for the whole alignment
    lat_ref = float(np.mean(np.concatenate([P_ll[:,0], Q_ll[:,0]])))
    lon_ref = float(np.mean(np.concatenate([P_ll[:,1], Q_ll[:,1]])))

    # 2) Convert to local meters
    P_xy = ll_to_en_batch(P_ll, lat_ref, lon_ref)
    Q_xy = ll_to_en_batch(Q_ll, lat_ref, lon_ref)

    # KD-tree on target points
    tree = cKDTree(Q_xy)

    t = np.zeros(2, dtype=float)
    history = []
    std_history = []
    error_history = []

    for _ in range(max_iters):
        # 3) Build correspondences: NN for each translated P
        P_t = P_xy + t
        dists, idx = tree.query(P_t)         # (n,)
        Q_match = Q_xy[idx]                  # (n,2)

        # Optional robust trimming
        # if trim_ratio > 0.0:
        keep = int(round((1.0 - trim_ratio) * len(dists)))
        sel = np.argpartition(dists, keep-1)[:keep]
        # else:
        #     sel = slice(None)

        # 4) Best translation for fixed matches = centroid difference
        t_star = Q_match[sel].mean(axis=0) - P_xy[sel].mean(axis=0)

        # 5) Update with damping
        t_new = (1.0 - damping) * t + damping * t_star

        # Record residual (mean NN distance)
        mean_resid = float(np.mean(dists)) if np.size(dists) else float('nan')
        history.append(mean_resid)
        std_history.append(np.std(dists))
        error_history.append(dists)

        # 6) Stop if translation change is tiny
        if np.linalg.norm(t_new - t) < tol_mm * 1e-3:
            t = t_new
            break

        t = t_new

    # --- Final pointwise errors after last translation ---
    P_final = P_xy + t
    final_dists, final_idx = tree.query(P_final)
    Q_match_final = Q_xy[final_idx]

    # Convert aligned source back to lat/lon
    P_fit_ll = en_to_ll_batch(P_final, lat_ref, lon_ref)
    Q_match_ll = en_to_ll_batch(Q_match_final, lat_ref, lon_ref)

    # Per-axis errors (meters)
    delta_e = (Q_match_final[:, 0] - P_final[:, 0]).tolist()
    delta_n = (Q_match_final[:, 1] - P_final[:, 1]).tolist()

    return {
        "t_xy": t,
        "P_fit_ll": P_fit_ll,
        "history": history,
        "std_history": std_history,
        "error_history": error_history,
        "lat_ref": lat_ref,
        "lon_ref": lon_ref,
        "final_dists_m": final_dists,
        "final_idx": final_idx,
        "P_fit_ll_pts": P_fit_ll,
        "Q_match_ll": Q_match_ll,
        "delta_e_m": delta_e,
        "delta_n_m": delta_n,
    }

def shift_gps_points(points, t_xy, lat_ref, lon_ref):
    tx, ty = t_xy
    lat_ref_rad = radians(lat_ref)

    # Compute lat/lon offset in degrees
    dlat = degrees(ty / R_EARTH)
    dlon = degrees(tx / (R_EARTH * cos(lat_ref_rad)))

    return [(lat + dlat, lon + dlon) for lat, lon in points]

def plot_icp_shift_arrow_inset(
    P_ll, P_fit_ll, Q_ll, t_xy, title="ICP Alignment of the Estimated Trajectory",
    inset_size="65%",
    inset_loc="upper right",
    inset_shift=(0, -1.25),
    half_window_deg=(1.2e-4, 9e-5),
    zoom_bbox=None,
    arrow_step=5,
    draw_arrows_on_main=False,
):
    def _format_axes_deg(ax_):
        # Disable scientific offset and format with 5 decimals
        ax_.ticklabel_format(axis='both', style='plain', useOffset=False)
        ax_.xaxis.set_major_locator(MaxNLocator(nbins=6))
        ax_.yaxis.set_major_locator(MaxNLocator(nbins=6))
        fmt = FuncFormatter(lambda v, pos: f"{v:.4f}")
        ax_.xaxis.set_major_formatter(fmt)
        ax_.yaxis.set_major_formatter(fmt)

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(Q_ll[:, 1], Q_ll[:, 0], 'b', lw=2, label="True Path")
    ax.plot(P_ll[:, 1], P_ll[:, 0], 'r--', label="Estimated Path")
    ax.plot(P_fit_ll[:, 1], P_fit_ll[:, 0], 'g', label="Shifted Estimated Path")

    tx, ty = t_xy
    txt = f"ΔE={tx:.2f} m, ΔN={ty:.2f} m, |Δt|={np.hypot(tx, ty):.2f} m"
    ax.add_artist(AnchoredText(txt, loc="upper left",  # <-- moved to top left
                               prop=dict(size=11), frameon=True))

    ax.set_xlabel("Longitude", fontsize=18)
    ax.set_ylabel("Latitude", fontsize=18)
    ax.set_xlim(-74.128, -74.1237)
    ax.set_aspect('equal', adjustable='box')
    ax.margins(0.02, 0.02)
    ax.grid(True, alpha=0.3)
    ax.set_title(title, fontsize=18)
    ax.legend()
    ax.tick_params(labelsize=13)

    _format_axes_deg(ax)

    dlon_all = P_fit_ll[:, 1] - P_ll[:, 1]
    dlat_all = P_fit_ll[:, 0] - P_ll[:, 0]

    if zoom_bbox is not None:
        lon_min, lon_max, lat_min, lat_max = zoom_bbox
        lon_min, lon_max = sorted([lon_min, lon_max])
        lat_min, lat_max = sorted([lat_min, lat_max])
        mask = (
            (P_ll[:, 1] >= lon_min) & (P_ll[:, 1] <= lon_max) &
            (P_ll[:, 0] >= lat_min) & (P_ll[:, 0] <= lat_max)
        )
        idx = np.nonzero(mask)[0]
        if len(idx) == 0:
            cand_idx = np.arange(0, len(P_ll), max(1, len(P_ll)//200))
            mag = np.hypot(dlon_all[cand_idx], dlat_all[cand_idx])
            i = cand_idx[np.argmax(mag)]
            idx = np.array([i])
        idx = idx[::max(1, arrow_step)]

        if len(idx) > 1:
            pad_lon = 0.05 * (lon_max - lon_min + 1e-12)
            pad_lat = 0.05 * (lat_max - lat_min + 1e-12)
            xlim = (lon_min - pad_lon, lon_max + pad_lon)
            ylim = (lat_min - pad_lat, lat_max + pad_lat)
        else:
            cx, cy = P_ll[idx[0], 1], P_ll[idx[0], 0]
            hx, hy = half_window_deg
            xlim = (cx - hx, cx + hx)
            ylim = (cy - hy, cy + hy)
    else:
        step = max(1, len(P_ll)//200)
        cand_idx = np.arange(0, len(P_ll), step)
        mag = np.hypot(dlon_all[cand_idx], dlat_all[cand_idx])
        i = cand_idx[np.argmax(mag)]
        idx = np.array([i])
        cx, cy = P_ll[i, 1], P_ll[i, 0]
        hx, hy = half_window_deg
        xlim = (cx - hx, cx + hx)
        ylim = (cy - hy, cy + hy)

    if draw_arrows_on_main and len(idx) > 0:
        ax.quiver(
            P_ll[idx, 1], P_ll[idx, 0],
            dlon_all[idx], dlat_all[idx],
            angles='xy', scale_units='xy', scale=1,
            width=0.006, headwidth=3, headlength=4, headaxislength=3,
            color='0.2', zorder=5
        )

    axins = inset_axes(ax, width=inset_size, height=inset_size,
                       loc=inset_loc,
                       bbox_to_anchor=(inset_shift[0], inset_shift[1], 1, 1),
                       bbox_transform=ax.transAxes,
                       borderpad=0.8)

    axins.plot(Q_ll[:, 1], Q_ll[:, 0], 'b', lw=1)
    axins.plot(P_ll[:, 1], P_ll[:, 0], 'r--', lw=1)
    axins.plot(P_fit_ll[:, 1], P_fit_ll[:, 0], 'g', lw=1)

    axins.quiver(
        P_ll[idx, 1], P_ll[idx, 0],
        dlon_all[idx], dlat_all[idx],
        angles='xy', scale_units='xy', scale=1,
        width=0.012, headwidth=3, headlength=3, headaxislength=3,
        color='k', zorder=6
    )

    axins.yaxis.tick_right()
    axins.yaxis.set_label_position("right")
    axins.set_xlim(*xlim)
    axins.set_ylim(*ylim)
    axins.set_aspect('equal', adjustable='box')
    axins.grid(True, alpha=0.2)

    _format_axes_deg(axins)
    axins.xaxis.set_major_locator(MaxNLocator(nbins=2, prune=None))
    axins.yaxis.set_major_locator(MaxNLocator(nbins=2, prune=None))

    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.3")
    return fig, ax

def plot_icp_with_all_matches(P_ll, Q_ll, trim_ratio=0.1, damping=0.8, tol_mm=0.2, max_iters=100,
                              show_original_source=True, zoom_bbox=None, save_prefix=None):
    """Run ICP, plot the shift-arrow inset figure and a zoomed match figure, print error stats."""
    P_ll = np.array(P_ll, dtype=float)
    Q_ll = np.array(Q_ll, dtype=float)

    # 1) Run ICP (translation-only) — returns everything we need
    icp = icp_translation_points_gps(
        P_ll, Q_ll,
        max_iters=max_iters,
        trim_ratio=trim_ratio,
        damping=damping,
        tol_mm=tol_mm,
    )
    t_xy      = icp["t_xy"]
    P_fit_ll  = icp["P_fit_ll"]
    dists     = icp["final_dists_m"]

    # 2) Shift-arrow inset plot
    if zoom_bbox is None:
        zoom_bbox = (-74.124843, -74.124297, 46.256379, 46.256168)

    fig, ax = plot_icp_shift_arrow_inset(
        P_ll, P_fit_ll, Q_ll, t_xy,
        inset_loc="upper right",
        inset_shift=(-0.2, 0),
        zoom_bbox=zoom_bbox,
        arrow_step=100,
        draw_arrows_on_main=False,
    )
    if save_prefix:
        fig.savefig(f"{save_prefix}_icp_shift.png", dpi=200, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

    # 3) Zoomed comparison plot
    fig2 = plt.figure(figsize=(8, 6))
    plt.plot(Q_ll[:, 1], Q_ll[:, 0], label="True line (Q)", linewidth=2)
    if show_original_source:
        plt.plot(P_ll[:, 1], P_ll[:, 0], linestyle="--", label="Original source (P)")
    plt.plot(P_fit_ll[:, 1], P_fit_ll[:, 0], label="Shifted source (P after ICP)", linewidth=2)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    if save_prefix:
        fig2.savefig(f"{save_prefix}_zoomed.png", dpi=200, bbox_inches='tight')
        plt.close(fig2)
    else:
        plt.show()

    # 4) Error statistics
    mean_err = float(np.mean(dists))
    std_err  = float(np.std(dists, ddof=1)) if len(dists) > 1 else 0.0
    rmse     = float(np.sqrt(np.mean(dists ** 2)))
    print(f"Error stats (m): mean={mean_err:.3f}, std={std_err:.3f}, RMSE={rmse:.3f}")

    return icp

if __name__ == "__main__":
    # --- Configuration ---
    estimation_path = ...
    estimated_positions = csv_to_id_dict(estimation_path)

    # --- Ground truth 1: GPSLOG_G_money ---
    true_path_1 = ...
    result_id_1 = 1.0
    subsample_1 = 10

    ground_truth_1 = parse_csv_to_lists(true_path_1)[::subsample_1]
    estim_1 = [list(pos) for pos in estimated_positions[result_id_1].values()]

    plot_icp_with_all_matches(
        np.array(estim_1), np.array(ground_truth_1),
        trim_ratio=0.1, damping=0.8, tol_mm=0.2, max_iters=100,
        save_prefix="output/gpslog_g",
    )

    # --- Ground truth 2: GPSLOG_R_money ---
    true_path_2 = ...
    result_id_2 = 2.0
    subsample_2 = 1

    ground_truth_2 = parse_csv_to_lists(true_path_2)[::subsample_2]
    estim_2 = [list(pos) for pos in estimated_positions[result_id_2].values()]

    plot_icp_with_all_matches(
        np.array(estim_2), np.array(ground_truth_2),
        trim_ratio=0.1, damping=0.8, tol_mm=0.2, max_iters=100,
        save_prefix="output/gpslog_r",
    )