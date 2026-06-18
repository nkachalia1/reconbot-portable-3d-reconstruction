# Hardware Runbook

## Deployed Architecture

The built-in laptop webcam cannot be physically attached to the Pi. The field
system therefore uses two networked nodes:

```text
Laptop webcam -> laptop camera service -> Wi-Fi -> Raspberry Pi coordinator
                                                -> session storage
                                                -> live dashboard
```

The laptop service owns webcam access, live preview, MP4 recording, and a
separate reconstruction worker. The Pi starts and stops recordings, copies
completed videos into the field session, records telemetry, proxies
reconstruction jobs, and serves the operator dashboard. COLMAP and OpenMVS
remain on the laptop.

## Local Simulation First

From Ubuntu/WSL:

```bash
cd "/mnt/c/Users/Neel/Documents/3D Scene Reconstruction"
python3 -m pip install -e ".[dashboard,dev]"
python3 scripts/local_field_demo.py
```

Open `http://127.0.0.1:5000`, choose **Field**, start a session, record a short
orbit, and stop the recording. This uses `data/frames/session_003` as a
deterministic virtual camera and writes the MP4 to
`data/field_sessions/demo/`.

## Laptop Camera Node

This service must run in native Windows Python because WSL does not normally
receive frames from the built-in laptop webcam. Check the installation:

```powershell
py --version
```

If that reports `No installed Python found`, install 64-bit Python 3.11 or
newer for Windows, including the Python launcher. Then create a separate
Windows environment; the WSL `.venv` cannot be reused:

```powershell
py -m venv .venv-win
.\.venv-win\Scripts\python -m pip install --upgrade pip
.\.venv-win\Scripts\python -m pip install -e ".[dashboard]"
```

Set a shared token and start the sensor service:

```powershell
$env:RECONBOT_TOKEN = "replace-with-a-long-random-token"
$env:RECONBOT_OPENMVS_BIN = "$HOME\Downloads\OpenMVS_Windows_x64\vc17\x64\Release"
.\.venv-win\Scripts\python scripts/laptop_camera_node.py `
  --host 0.0.0.0 --port 5001 --camera-index 0
```

This command also starts the reconstruction worker on port `5002`. Add
`--no-reconstruction-worker` only when intentionally running capture by itself.

Confirm locally:

```powershell
Invoke-RestMethod http://127.0.0.1:5001/api/health
```

Find the laptop IPv4 address with `ipconfig`. If Windows Firewall blocks the
Pi, allow inbound TCP port 5001 on the private network:

```powershell
New-NetFirewallRule -DisplayName "ReconBot Camera Node" `
  -Direction Inbound -Protocol TCP -LocalPort 5001,5002 -Action Allow -Profile Private
```

## Raspberry Pi Setup

Copy the project to `/home/pi/portable-recon-robot`, then:

```bash
cd /home/pi/portable-recon-robot
sudo apt update
sudo apt install -y python3-venv python3-numpy
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install "flask>=3,<4"
python -m pip install -e . --no-deps
```

Build `dashboard/dist` on the laptop and copy that directory to the same path
on the Pi. Then test the coordinator interactively:

```bash
export RECONBOT_CAMERA_URL=http://LAPTOP_IPV4:5001
export RECONBOT_RECONSTRUCTION_URL=http://LAPTOP_IPV4:5002
export RECONBOT_TOKEN=replace-with-a-long-random-token
python scripts/pi_coordinator.py
```

Open `http://raspberrypi.local:5000` from the laptop or a phone on the same
Wi-Fi network. Keep this operator API on a trusted private network.

## Install The Pi Service

The installer fills in the current Pi username and project directory when it
creates the systemd unit.

```bash
chmod +x scripts/install_pi_service.sh
./scripts/install_pi_service.sh
sudo nano /etc/reconbot/coordinator.env
sudo systemctl restart reconbot-coordinator
sudo systemctl status reconbot-coordinator
```

Logs:

```bash
journalctl -u reconbot-coordinator -f
```

## Field Procedure

1. Charge the power bank and connect the Pi.
2. Put the Pi and laptop on the same Wi-Fi network.
3. Start the laptop camera node and verify `/api/health`.
4. Open the Pi dashboard and check that **Camera online** appears.
5. Start a named session and choose the intended arc direction.
6. Click **Start recording**, physically carry the laptop/webcam, and walk one
   smooth orbit around the target.
7. Record for 45 to 90 seconds while maintaining 60 to 80 percent overlap.
8. Click **Stop recording** and confirm that playback appears.
9. Click **Reconstruct video** and monitor the stage and percentage readout.
10. When processing completes, the dashboard opens the new model and adds it
    to Reconstruction history.

## Automated Reconstruction

The Windows worker performs the existing manual pipeline automatically:

```text
video -> sharp frames -> WSL COLMAP sparse SfM -> image undistortion
      -> Windows OpenMVS dense cloud -> 300k-face mesh -> texture -> GLB
```

Published artifacts are stored under:

```text
data/reconstruction_library/<session>/video.mp4
data/reconstruction_library/<session>/model.glb
data/reconstruction_library/<session>/metrics.json
data/reconstruction_library/<session>/pipeline.log
```

Successful jobs remove their multi-gigabyte intermediate depth maps to protect
laptop storage. Failed jobs retain the work directory and pipeline log for
diagnosis. The older `scripts/fetch_field_video.py` command remains available
as a manual recovery path.

## Capture Rules

- Move in a slow arc with 60 to 80 percent adjacent-frame overlap.
- Keep the target and background stationary while the camera changes position.
- Do not rotate a person, chair, or object in front of a fixed webcam. Standard
  SfM requires camera translation through a rigid scene.
- Add sideways baseline; do not rotate in place.
- Avoid glossy, transparent, blank, or moving targets.
- Keep exposure and subject distance stable.
- Use a wide, smooth orbit rather than standing still and panning.

## Operational Limits

- Stop or cool the Pi above 80 C.
- Keep at least 2 GB of session storage free.
- A dropped Wi-Fi request must not corrupt the current session.
- Dense reconstruction remains on the laptop because it exceeds the Pi's
  practical thermal and runtime budget.

## Interview Talking Point

The Pi is a field coordinator, not a token compute target. The split exposes a
real robotics design decision: sensor access, network reliability, storage,
power, thermals, operator feedback, and offline compute are separate concerns.
