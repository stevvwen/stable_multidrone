import warnings

import cv2
import numpy
import torch
import os

from ultralytics import YOLO
import gps_btracker
from utils.gps_util import positional_estimate
import pandas as pd
from datetime import datetime, timedelta
from utils.util import EKF_CV2D
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def draw_line(image, x1, y1, x2, y2, index, frame_id):
    w = 10
    h = 10
    color = (200, 0, 0)
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 200, 0), 2)
    # Top left corner
    cv2.line(image, (x1, y1), (x1 + w, y1), color, 3)
    cv2.line(image, (x1, y1), (x1, y1 + h), color, 3)

    # Top right corner
    cv2.line(image, (x2, y1), (x2 - w, y1), color, 3)
    cv2.line(image, (x2, y1), (x2, y1 + h), color, 3)

    # Bottom right corner
    cv2.line(image, (x2, y2), (x2 - w, y2), color, 3)
    cv2.line(image, (x2, y2), (x2, y2 - h), color, 3)

    # Bottom left corner
    cv2.line(image, (x1, y2), (x1 + w, y2), color, 3)
    cv2.line(image, (x1, y2), (x1, y2 - h), color, 3)

    text = f'ID:{str(index)}'
    cv2.putText(image, text,
                (x1, y1 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255),
                thickness=3, lineType=cv2.LINE_AA)

    cv2.putText(image, f'Frame: {frame_id}', (x1, y2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def save_gps_dict_to_csv(gps_dict, filename="gps_data.csv"):
    rows = []
    for key, points in gps_dict.items():
        for lat, lon, time in points:
            rows.append({"ID": key, "time": time, "Latitude": lat, "Longitude": lon})
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"Saved {filename} with {len(rows)} rows.")


def load_gps_dict(csv_path):
    df = pd.read_csv(csv_path)
    df['time'] = df['time'].astype(str)
    return {
        row['time']: row.drop(labels='time').tolist()
        for _, row in df.iterrows()
    }


def process_camera(model, device, frame, cam_id, gps_dict, current_time,
                   cam_boxes, cam_confidences, cam_object_classes,
                   cam_gps_estimates, bytetrack, writer):
    results = model.predict(
        source=frame,
        conf=0.001,
        iou=0.7,
        classes=[0],
        device=0 if device == "cuda" else None,
        verbose=False

    )

    drone_lat, drone_lon, drone_alt, drone_angle, drone_heading, camera_angle = gps_dict[current_time]

    if len(results):
        r = results[0]
        xyxy = r.boxes.xyxy
        conf = r.boxes.conf
        cls = r.boxes.cls

        if xyxy is not None and len(xyxy):
            xyxy = xyxy.detach().cpu().numpy().astype(int)
            conf = conf.detach().cpu().numpy()
            cls = cls.detach().cpu().numpy()
            for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):                            
                cam_boxes[cam_id].append([x1, y1, x2, y2])
                cam_confidences[cam_id].append(float(c))
                cam_object_classes[cam_id].append(int(k))
                cam_gps_estimates[cam_id].append(
                    positional_estimate(drone_lat, drone_lon, drone_alt,
                                        drone_angle, drone_heading,
                                        camera_angle, (x1+x2)/2, (y1+y2)/2))

    outputs = bytetrack.update(
        numpy.array(cam_boxes[cam_id]),
        numpy.array(cam_confidences[cam_id]),
        numpy.array(cam_object_classes[cam_id]),
        numpy.array(cam_gps_estimates[cam_id]), cam_id)

    if len(outputs) > 0:
        tracked_boxes = outputs[:, :4]
        identities = outputs[:, 4]
        tracked_classes = outputs[:, 6]
        for i, box in enumerate(tracked_boxes):
            if tracked_classes[i] != 0:  # 0 is for aqua
                continue
            x1, y1, x2, y2 = list(map(int, box))
            index = int(identities[i]) if identities is not None else 0
            draw_line(frame, x1, y1, x2, y2, index, bytetrack.frame_id)

        writer.write(frame.astype('uint8'))

    return outputs


def main():
    # --- Configuration: edit these lists to add/remove drones ---
    fly_records = [
        ...
    ]
    vid_paths = [
       ...
    ]
    model_path = "utils/yolo_drone.pt"
    start_time = "09:40.0"
    start_counter = 30
    frame_cutoff = 7151
    output_prefix = "output/merge4"

    num_cameras = 3

    # Load telemetry dicts
    time_dicts = [load_gps_dict(record) for record in fly_records]

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    model = YOLO(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    # Open readers and writers for each drone
    readers = [cv2.VideoCapture(p) for p in vid_paths]
    fps = int(readers[0].get(cv2.CAP_PROP_FPS))
    width = int(readers[0].get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(readers[0].get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    effective_fps = fps if fps > 0 else 30
    writers = [
        cv2.VideoWriter(f'{output_dir}/{name}.mp4', fourcc, effective_fps, (width, height))
        for name in [os.path.splitext(os.path.basename(p))[0] for p in vid_paths]
    ]

    start_t = datetime.strptime(start_time, "%M:%S.%f")

    bytetrack = gps_btracker.BYTETracker(num_cameras=num_cameras, time_lost=60)

    tracked_keys = {}
    aquas_ekalman_weights = {}
    aquas_ekalman_positions_weights = {}
    counter = 0

    while readers[0].isOpened():
        reads = [r.read() for r in readers]
        frames = [f[1] for f in reads]

        if bytetrack.frame_id == frame_cutoff:
            break

        if counter % 3 == 0 and counter > start_counter:
            current_time = start_t.strftime("%M:%S.%f")[:-5]
            print(f"\nProcessing frame at time: {current_time}, {bytetrack.frame_id + 1}")

            cam_boxes = [[] for _ in range(num_cameras)]
            cam_confidences = [[] for _ in range(num_cameras)]
            cam_object_classes = [[] for _ in range(num_cameras)]
            cam_gps_estimates = [[] for _ in range(num_cameras)]

            merged_results = []
            for cam_id in range(num_cameras):
                outputs = process_camera(
                    model, device, frames[cam_id], cam_id,
                    time_dicts[cam_id], current_time,
                    cam_boxes, cam_confidences, cam_object_classes,
                    cam_gps_estimates, bytetrack, writers[cam_id]
                )
                merged_results += outputs.tolist()

            # -- Confidence-weighted GPS fusion + EKF --
            yolo_weights = {}
            for row in merged_results:
                key = row[4]
                yolo_weights[key] = yolo_weights.get(key, 0) + row[5]

            weighted_estimates = {}
            for row in merged_results:
                key = row[4]
                w = row[5] / yolo_weights[key]
                lat_w = row[8] * w
                lon_w = row[9] * w
                if key not in weighted_estimates:
                    weighted_estimates[key] = [lat_w, lon_w]
                else:
                    weighted_estimates[key][0] += lat_w
                    weighted_estimates[key][1] += lon_w

            print(f"weighted estimates: {weighted_estimates.keys()}")
            for track_id, pos in weighted_estimates.items():
                if track_id not in tracked_keys:
                    tracked_keys[track_id] = [pos]
                    ekf = EKF_CV2D(dt=0.1, q_acc=0.5, use_latlon_measurements=True)
                    ekf.step((pos[0], pos[1]))
                    aquas_ekalman_weights[track_id] = ekf
                    aquas_ekalman_positions_weights[track_id] = [[pos[0], pos[1], current_time]]
                else:
                    tracked_keys[track_id].append(pos)
                    aquas_ekalman_weights[track_id].predict()
                    ekf_out = aquas_ekalman_weights[track_id].step((pos[0], pos[1]))
                    aquas_ekalman_positions_weights[track_id].append([ekf_out[0], ekf_out[1], current_time])

        if counter % 3 == 0:
            start_t += timedelta(milliseconds=100)

        counter += 1

        if bytetrack.frame_id % 1000 == 0 and bytetrack.frame_id > 0:
            bytetrack.save_trajectories_csv(output_prefix)

    print("Saving trajectories...")
    bytetrack.save_trajectories_csv(output_prefix)

    # -- Save EKF-fused GPS estimates & plot --
    save_gps_dict_to_csv(aquas_ekalman_positions_weights, f"{output_prefix}_weighted.csv")

    plt.figure(figsize=(8, 6))
    for key, lists in aquas_ekalman_positions_weights.items():
        lats = [s[0] for s in lists]
        lons = [s[1] for s in lists]
        plt.plot(lons, lats, label=key)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("EKF Weighted GPS Estimates")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_prefix}_weighted_plot.png")
    print(f"Saved plot to {output_prefix}_weighted_plot.png")

    for r in readers:
        r.release()
    for w in writers:
        w.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
