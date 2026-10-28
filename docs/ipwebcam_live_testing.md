# Live IP Webcam testing (advisory-only)

This is the P.R.I.S.M. setup guide for the live IP Webcam probe. The repository
root `README.md` describes the historic PRISM prototype; it is not the
P.R.I.S.M. integration guide.

The runner connects to the Android **IP Webcam** app's MJPEG endpoint, captures
a bounded number of frames, runs the existing LibreYOLO9t 2-D detector and
image-space tracker, writes an annotated recording, and audits the existing ORB
/ fundamental-matrix front end on the recorded clip. It is an advisory
experiment, not a navigation or hazard-accuracy validation.

## 1. Prerequisites

1. Install **IP Webcam** on the OnePlus Nord 4 and start its server.
2. Connect the phone and the PC to the same hotspot or Wi-Fi network.
3. Read the address shown on the phone. It changes when the hotspot restarts.
4. The default stream address in P.R.I.S.M. is:

   ```text
   http://10.78.70.188:8080/video
   ```

Use plain `http://`. Do not switch to `https://` unless SSL was explicitly
configured on the IP Webcam server.

## 2. Check connectivity before blaming the code

On the PC:

1. Open the dashboard in a browser: `http://10.78.70.188:8080/`.
2. Check the TCP port from PowerShell:

   ```powershell
   Test-NetConnection 10.78.70.188 -Port 8080
   ```

If the dashboard does not open, this is a phone/PC network or IP Webcam server
problem, not a P.R.I.S.M. bug. Confirm the IP Webcam server is running, the IP
and port are current, and no firewall blocks the connection.

## 3. Run the probe

From the repository root, using the project virtual environment:

```powershell
.venv\Scripts\python.exe tools\run_prism_ipwebcam.py
```

A bounded run with more frames:

```powershell
.venv\Scripts\python.exe tools\run_prism_ipwebcam.py --max-frames 60
```

Override the address without editing the tool:

```powershell
.venv\Scripts\python.exe tools\run_prism_ipwebcam.py --url http://10.78.70.188:8080/video
```

Or use the environment variable:

```powershell
$env:PRISM_IPWEBCAM_URL = "http://10.78.70.188:8080/video"
.venv\Scripts\python.exe tools\run_prism_ipwebcam.py
```

Address precedence is: explicit `--url`, then `PRISM_IPWEBCAM_URL`, then the
default above. A base URL such as `http://10.78.70.188:8080` is completed with
`/video`; a URL that already ends in `/video` is left alone and never doubled.

Press `Ctrl+C` to stop early. The capture is released and the report is written
before the process exits.

Useful bounded options:

| Option | Meaning |
| --- | --- |
| `--max-frames N` | Frame budget, 1-3600 (default 30). Never unbounded. |
| `--open-retries N` | Extra open attempts after the first (default 2). |
| `--read-retries N` | Consecutive read retries before stopping (default 3). |
| `--open-timeout-ms` / `--read-timeout-ms` | Bounded OpenCV/FFmpeg timeouts. |
| `--fps` | Writer FPS used only when the stream reports none (default 10). |
| `--weights` | Local LibreYOLO9t checkpoint; nothing is downloaded. |
| `--output` | Exact new output directory; existing directories are refused. |
| `--output-root` | Root for a unique timestamped run directory. |

## 4. Outputs

Each run creates a unique directory under `output/ipwebcam_runs/`, or exactly
`--output` when supplied. Prior runs are never overwritten.

| File | Contents |
| --- | --- |
| `run_report.json` | Configuration, connection diagnostics, frame/latency statistics, detector identity and checkpoint hash, limitations. |
| `frame_events.jsonl` | Per-frame timestamp, read gap, detector status, detection count, track count. |
| `detections.json` | Per-frame 2-D detections and provenance, or the detector error. |
| `tracks.csv` | Image-space tracks, including explicitly labelled predictions. |
| `camera_motion.csv` | ORB/fundamental-matrix diagnostics over the recorded clip. |
| `capture.mp4` | Bounded raw recording. |
| `annotated_capture.mp4` | Recording with 2-D boxes, track IDs, and status. |
| `README.md` | Short guide to the run outputs. |

Exit codes: `0` completed, `2` connection/configuration failure, `3` no frames
or an interrupted stream, `4` processing error, `130` interrupted by the user.

## 5. Connection diagnostics

The runner uses bounded open/read retries and never reconnects in an endless
loop. It does not duplicate a previous frame after a failed read.

| Symptom | Meaning |
| --- | --- |
| Dashboard does not open; TCP probe fails | Phone/server unreachable: same hotspot, server off, stale IP/port, or firewall. |
| TCP reaches the phone but the stream does not open | The `/video` endpoint or MJPEG codec is unavailable; check the URL and OpenCV/FFmpeg. |
| Stream opens but no frame decodes | The endpoint is not the MJPEG stream, or the read timeout is too short. |
| `https://` failure | IP Webcam usually serves plain HTTP. Use `http://` unless SSL is configured. |
| Detector unavailable | The local checkpoint is missing or the detector dependency failed; capture continues without detections. This is reported, never filled in. |

## 6. Limitations

- No command is sent to a drone, flight controller, motor, or vehicle.
- Detections are 2-D COCO image observations, not hazards, depth, clearance, or
  traversability. An empty detection list does not prove a path is safe.
- Visual motion is diagnostic. Without validated intrinsics and calibration it
  is tracking-only, and monocular relative motion has arbitrary scale. It is not
  metric odometry, GPS-denied localization, or navigation readiness.
- MJPEG frames carry no hardware timestamp. Frames are timestamped when read,
  and read gaps are recorded; buffered frames may be older than their processing
  time.
- No terrain, mapping, obstacle-avoidance, or hazard accuracy is validated by
  this probe.

## 7. Tests

```powershell
.venv\Scripts\python.exe -m pytest tests/test_ipwebcam_runner.py -q
```

The automated tests mock video capture and TCP probes; they do not require a
phone. A physical-camera connection is **not** verified by the test suite. Run
the command in section 3 against the phone to verify the live stream, and read
`run_report.json` for what was actually measured.
