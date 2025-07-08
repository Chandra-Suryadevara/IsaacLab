# Neck-Injection Area Demo (Detectron2)

This project shows how to run a Detectron2 model that locates the optimal neck‑injection area in cattle video frames, smooths detections with a median filter, and writes the result to a new MP4.

---

## 1 · Quick start (Windows + CUDA 12.1)

```powershell
# ❶ Clone and enter the repo
git clone https://github.com/<your‑user>/neck-detection.git
cd neck-detection

# ❷ Create the Conda environment
conda env create -f environment.yml
conda activate detectron2

# ❸ Download the model weights (≈ 200 MB)
# Replace the URL with your real OneDrive share link
powershell -Command "Invoke-WebRequest -Uri 'https://1drv.ms/u/s!/<share-id>/model_final.pth' -OutFile model_final.pth"

# ❹ (Optional) Set PYTHONPATH if Detectron2 lives in a sibling folder
# set PYTHONPATH=%CD%\detectron2_src

# ❺ Run the demo
python process_neck_video_no_fallback.py ^
    --config-file config.yaml ^
    --input demo.mp4 ^
    --output output\demo_processed.mp4 ^
    --confidence-threshold 0.85 ^
    --opts MODEL.WEIGHTS model_final.pth
```

The batch file `run_video.bat` automates steps ❸–❺ once the environment is ready.

---

## 2 · Repository layout

```
neck-detection/
│
├─ environment.yml            ← Python/Conda dependencies
├─ config.yaml                ← Detectron2 config
├─ process_neck_video_no_fallback.py
├─ run_video.bat              ← Windows helper
├─ README.md
└─ model_final.pth            ← **Downloaded** (not stored in Git)
```

The model file is **not** committed to this repository; download it once with the command above or grab it manually from the OneDrive link.

---

## 3 · Prerequisites

| Component                  | Why                                                    | Where to get                                               |
| -------------------------- | ------------------------------------------------------ | ---------------------------------------------------------- |
| GPU driver + **CUDA 12.1** | Matches `pytorch-cuda=12.1` in `environment.yml`.      | NVIDIA driver download                                     |
| Visual Studio Build Tools  | Required when Detectron2 builds from source.           | [https://aka.ms/vsbuildtools](https://aka.ms/vsbuildtools) |
| **ffmpeg** (optional)      | Extra codecs if OpenCV can’t write your target format. | [https://ffmpeg.org/](https://ffmpeg.org/)                 |

Everything else is installed by Conda or pip.

---

## 4 · Troubleshooting

| Symptom                            | Fix                                                                                           |
| ---------------------------------- | --------------------------------------------------------------------------------------------- |
| `ImportError: DLL load failed`     | CUDA mismatch—install a driver ≥ 12.1 or change the `pytorch-cuda` version.                   |
| `ModuleNotFoundError: detectron2`  | Make sure **one** Detectron2 line in `environment.yml` is uncommented, then recreate the env. |
| Output video is empty / zero bytes | Check that `fps` detected by OpenCV isn’t zero. Pass `--fps 30` to the script if needed.      |

---

## 5 · License

MIT License unless stated otherwise.

