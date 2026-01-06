# Raspberry Pi 5 Portable Stereo Vision System (Power PCB + UI)

A low-cost portable stereo vision system based on **Raspberry Pi 5** and dual cameras.
This repository documents my main contributions to the project:

- **Custom Power PCB**: stable multi-rail power delivery for Pi 5 + peripherals, with protection and validation
- **On-device UI**: local control panel for preview/recording/parameters/status, designed for portable use

---

## What I Built

![PCB_3D_View](path/to/image.png)
### 1) Power PCB (Primary Contribution)
A custom power board designed to support a portable Pi 5 stereo vision setup.

**Goals**
- Provide stable power rails for Raspberry Pi 5 and accessories (camera(s), display, storage)
- Handle battery input safely with protection
- Support portable operation (efficiency, thermal, noise)

**Highlights**
- Power architecture: `Battery/Adapter → DC/DC → 5V rail(s) → Pi 5 + peripherals`
- Protection: fuse / reverse polarity / TVS (as applicable)
- Validation: load test, ripple/noise check, thermal observation

**Where to look**
- `hardware/power_pcb/` for schematics, PCB, BOM, assembly notes, and test logs.

---

### 2) UI (Primary Contribution)
A lightweight on-device UI for operating the stereo vision system.

**Core UI features**
- Live preview window (rectified view / overlay view if available)
- Start/Stop recording
- Adjustable runtime parameters (exposure, resolution, FPS, stereo params if exposed)
- System status panel (FPS, CPU temp, storage, battery/voltage if available)
- Simple “profile/config” loading

**Where to look**
- `ui/` for UI source code, assets, and packaging scripts
- `docs/ui.md` for screenshots and user flow

---

## Repository Structure

```text
.
├─ hardware/
│  └─ power_pcb/
│     ├─ schematics/            # PDF exports / design notes
│     ├─ pcb/                   # board files (Altium/KiCad exports)
│     ├─ bom/                   # BOM + sourcing notes
│     ├─ fabrication/           # gerbers, drill, pick&place (if available)
│     ├─ bringup/               # bring-up checklist + measurement logs
│     └─ test_results/          # ripple/thermal/load test data
├─ ui/
│  ├─ app/                      # UI code (Qt/QML or other)
│  ├─ assets/                   # icons, images, fonts
│  ├─ configs/                  # UI config / presets
│  └─ scripts/                  # run/build scripts
├─ integration/
│  ├─ api_contract.md           # UI ↔ vision pipeline interface assumptions
│  └─ sample_data/              # sample frames / json for UI testing
├─ docs/
│  ├─ system_overview.md
│  ├─ power_pcb.md              # design decisions + test summary
│  ├─ ui.md                     # UI workflow + screenshots
│  └─ troubleshooting.md
└─ references/
   └─ stereo_pipeline_notes.md  # brief context (owned by other team members)
