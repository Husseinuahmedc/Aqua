# AquaRescue

AquaRescue is a local desktop **Search & Rescue Analysis Tool** for inspecting UAV RGB images and videos from flood and water environments using traditional Digital Image Processing (DIP). Its primary **Rescue Analysis** workspace highlights **Potential Person**, **Floating Object**, and **Unknown Candidate** regions for human review, displays their visual measurements and classification reasoning, supports frame-by-frame video candidate analysis and experimental tracking, and exports review records to CSV and JSON. Supporting **Water Analysis** and **Evaluation** workspaces provide configurable HSV potential-water segmentation, video coverage history, and manual reference-mask comparison. Candidate outputs are heuristic suggestions requiring human verification, not confirmed person or survivor detections.

> **Search & Rescue pipeline guide:** [docs/PIPELINE_README.md](docs/PIPELINE_README.md) — architecture, DIP methods, data adapters, application operation, video tracking, evaluation protocol, configuration, team responsibilities, tests, and limitations.

## Requirements

- Python 3.12 or newer
- Git
- The approved Python packages listed in `requirements.txt`

`requirements.txt` records the six approved direct dependencies. `requirements-lock.txt` pins the resolved environment, including their required transitive packages, for repeatable installation. The lock snapshot is generated and verified on Linux with Python 3.14; installation and packaging still need independent Windows verification.

## Create and activate a virtual environment

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python --version
```

Confirm the reported Python version is 3.12 or newer.

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
```

### Windows Command Prompt

```bat
py -3 -m venv .venv
.venv\Scripts\activate.bat
python --version
```

Confirm the reported Python version is 3.12 or newer in either Windows shell.

## Install and verify dependencies

With the virtual environment activated:

```bash
python -m pip install -r requirements-lock.txt
python -m pip check
python -m pytest
```

The documented/tested environment for this MVP is Linux with CPython 3.14.7. The application requires Python 3.12 or newer; other supported Python/platform combinations still need their own verification.

## Launch AquaRescue

With the virtual environment activated, from the project root run:

```bash
python -m app.main
```

The application opens directly in **Rescue Analysis**:
1. Import a local UAV image (`Open image`) or video (`Open video`).
2. Click **Analyze** to run traditional DIP candidate generation (advanced settings are collapsed by default with sensible defaults active).
3. Compare the original input beside annotated **Potential Person**, **Floating Object**, and **Unknown Candidate** regions; for videos, switch between all frame candidates, the current frame, or persistent experimental tracks.
4. Select any candidate to highlight its bounding region on the image/frame, read its DIP reasoning and visual measurements, and update its review decision (`Requires Human Review`, `Confirmed Candidate`, or `False Alarm`).
5. Export candidate review records and provenance to CSV or JSON.

Supporting water-segmentation tools remain available in **Water Analysis** (with Still Image and Video Frame modes), and **Evaluation** compares a predicted water mask with a manually annotated binary reference mask. No dataset is downloaded automatically.

## Design system

The desktop visual system is documented in [`DESIGN.md`](DESIGN.md). Its 18 named base color tokens are implemented in `app/theme.py`; `.impeccable/design.json` records the token ramps and component references. Ramp colors are design references; the application uses the named base tokens.

Masks report pixel counts and pixel percentages only; they do not represent geographic area. HSV thresholding can include reflections, shadows, roofs, and permanent water, and can miss floodwater. The unchanged default baseline has been evaluated against the 700-pair Floodwater manual set; see [Testing Strategy](docs/TESTING_STRATEGY.md) for the protocol, per-image files, aggregate results, and failure review. This is visible-water mask agreement, not floodwater classification or field validation.

The Search & Rescue workspace produces traditional-DIP candidate regions for human review only. Category names and rule scores are heuristics, not confirmed person/object detection or probabilities. SeaDronesSee has not been acquired for evaluation, NITC Person Rescue remains permission-gated, and Floodwater labels water/background only. The Linux implementation and environment tests are exercised locally. Windows instructions are provided above, but Windows installation, codecs, and packaging have not been verified.

To update approved package versions, edit only the six direct dependencies in `requirements.txt`, resolve them in a clean supported Python environment, then regenerate `requirements-lock.txt` with `python -m pip freeze --local`. Review the resolved changes before using the updated lock.

## Current status and limitations

The app has a working traditional HSV threshold baseline and manual-mask metric calculation. Its evaluation on the full manual subset is documented, but thresholds remain scene-dependent: shadows, reflections, roofs, and permanent water can be included, and floodwater can be missed. Pixel coverage is not geographic area. Candidate regions are experimental geometry proposals for human inspection only. Video codec support depends on the local OpenCV build.
