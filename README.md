# Falling prediction webcam MVP

Windows CPU MVP using OpenVINO `human-pose-estimation-0001`, the Intel Open
Model Zoo OpenPose PAF association decoder, and an OpenCV overlay. This is a
demonstration, not a medical device.

```powershell
uv sync
python scripts/download_model.py
uv run falling-prediction --model models/human-pose-estimation-0001/human-pose-estimation-0001.xml --device CPU
```

Use `--bed-left`, `--bed-top`, `--bed-right`, and `--bed-bottom` for a normalized
rectangular bed. Press **Esc** to stop. Model XML/BIN files are not committed.
The downloader retrieves the official 2023.0 FP32 IR and verifies its SHA-384
checksums. The model contract is input `[1,3,256,456]` BGR, PAF output
`[1,38,32,57]`, and heatmap output `[1,19,32,57]`. The decoder performs
heatmap max-pool NMS, all 19 OpenPose limb associations, PAF connection
suppression and pose merging, then returns COCO poses `(N,17,3)` and person
scores `(N,)`. Intel attribution and the full Apache-2.0 notice are in
`THIRD_PARTY_NOTICES.md`.

When no person is detected the temporal risk history is reset and the overlay
shows `WAITING`; an absent pose is never displayed as `SAFE`.
