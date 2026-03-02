# Stable Multi-Drone GNSS Tracking System for Marine Robots

This repository contains the official implementation of the paper **"Stable Multi-Drone GNSS Tracking System for Marine Robots,"** accepted at the **2026 IEEE International Conference on Robotics & Automation (ICRA)**.

Our system provides a robust framework for tracking marine robots across multiple drones by fusing vision-based detections with GNSS telemetry data. We extend the BYTETracker algorithm with a custom GPS Kalman Filter and multi-drone confidence-weighted sensor fusion to reliably estimate target coordinates in real-time.


## ⚙️ Installation



```bash
git clone https://github.com/stevvwen/stable_multidrone.git
cd stable_multidrone
conda create -n multidrone python=3.11 
conda activate multidrone
pip install -r requirements.txt
```



## 🚀 Quick Start

The main pipeline for running the multi-drone tracking and fusion is implemented in `stable_multidrone_m4.py`.

1. **Configure Paths:** Open `stable_multidrone_m4.py` and modify the configuration lists inside the `main()` function to point to your local video files and telemetry CSVs:
* `drone_names`: Names of the drones.
* `fly_records`: Paths to the telemetry CSV files for each drone.
* `vid_paths`: Paths to the corresponding drone video feeds (e.g., MP4 format).
* `model_path`: Ensure you have the trained YOLOv8 weights placed at `utils/yolo_drone.pt`.

2. **Download the data:** Please visit [AMP2026](https://huggingface.co/datasets/edwinmeriaux/AMP2026/tree/main/Tracking) and go to `Multi-Drone Tracking of submerged Robotic Platforms` folder and download everything from the subdirectories.


2. **Run the Tracker:**
```bash
python stable_multidrone_m4.py
```


The output videos with bounding boxes and trajectory overlays, along with the fused GNSS estimates (`merge4_weighted.csv`) and plots, will be saved into an `output/` directory automatically.

## 📂 Error Calculation

The `plotting.py` script evaluates the tracking system's accuracy. It utilizes Iterative Closest Point (ICP) alignment to map the estimated trajectory to the ground-truth GNSS logs, compensating for static offsets. It then computes the residual errors and generates visualization plots.

### 1. Format Your Ground Truth Data

Ensure your ground-truth GNSS CSV files contain the columns `lat_decimal` and `lon_decimal`. The estimated tracking output (e.g., `merge4_weighted.csv`) is formatted automatically by the tracker.

### 2. Configure the Script

Open `plotting.py` and navigate to the `if __name__ == "__main__":` block at the bottom of the file. Update the following configuration variables to match your data:

* **`estimation_path`**: Path to the fused tracker output (e.g., `"output/merge4_weighted.csv"`).
* **`true_path_1`** / **`true_path_2`**: Paths to your ground-truth CSV files.
* **`result_id_1`** / **`result_id_2`**: The integer ID assigned to the tracked object in the tracker output (e.g., `1.0` or `2.0`).
* **`subsample_1`** / **`subsample_2`**: Step size for subsampling the ground-truth data to match the estimated trajectory's frequency.
* **`trim_to_1`**: An optional integer to slice the array if the ground truth log runs longer than the video recording.

### 3. Run the Evaluation

```bash
python plotting.py
```

### 4. Outputs

* **Console Metrics:** The script will print the Mean Error, Standard Deviation, and Root Mean Square Error (RMSE) in meters to your terminal.
* **Visualizations:** Generates and saves high-resolution plots to the specified `save_prefix` path:
* `*_icp_shift.png`: An overview map showing the true path, the original estimated path, and the ICP-shifted path with an inset zoom.
* `*_zoomed.png`: A close-up view comparing the alignment of the estimated trajectory against the ground truth.



## 📖 Citation

If you find our work useful in your research, please consider citing our ICRA 2026 paper:

```bibtex
@article{wen2025stable,
  title={Stable Multi-Drone GNSS Tracking System for Marine Robots},
  author={Wen, Shuo and Meriaux, Edwin and Guzm{\'a}n, Mariana Sosa and Wang, Zhizun and Shi, Junming and Dudek, Gregory},
  journal={IEEE International Conference on Robotics and Automation (ICRA) 2026},
  year={2025}
}

```
