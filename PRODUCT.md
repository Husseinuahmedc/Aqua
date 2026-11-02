# Product

<!-- impeccable:product-schema 1 -->

## Platform

adaptive

Cross-platform native desktop application for Linux and Windows. Adapt operating-system interactions where appropriate while retaining a consistent scientific-analysis workspace; this product is not a mobile application.

## Stack

User-approved: Python 3.12+, PySide6, OpenCV, NumPy, Pandas, Matplotlib, and Pytest. See `AGENTS.md` and `requirements.txt` for binding dependency rules and versions.

## Users

Primary users are research and evaluation practitioners (including researchers and students) examining aerial flood-related imagery with traditional image-processing methods. They need to inspect source imagery, understand and tune processing parameters, compare masks with manual annotations, and reproduce measurements. Field imagery reviewers are a secondary audience.

## Product Purpose

AquaRescue is a local desktop aid for inspecting RGB drone images and videos captured in flood situations. It applies interpretable digital image processing to identify potential water regions, summarize frame-by-frame mask coverage/change, compare predictions with manually annotated masks, and help users review candidate regions. Success means making that analysis more inspectable and reproducible while keeping the human responsible for interpretation.

## Positioning

The product presents configurable, traditional DIP results and their limitations for human inspection. It is not an autonomous rescue, alerting, or decision system and does not claim reliable floodwater classification or semantic object detection.

## Operating Context

Users work with local RGB aerial images and video, often comparing an original frame, a potential-water mask, and a manually selected reference mask. Processing is offline and may take long enough to require progress and cancellation. Video is handled frame by frame. The approved deployment targets are Linux and Windows; codec support can vary by system.

## Capabilities and Constraints

- The V1 target includes local image/video inspection, traditional preprocessing and HSV threshold segmentation, morphology/contour analysis, frame summaries, mask evaluation, and explicit export. Use `docs/PROJECT_SPECIFICATION.md` and `docs/DEVELOPMENT_ROADMAP.md` for the authoritative live implementation status.
- Candidate-region extraction is experimental and only proposes areas for human inspection. It must never be described as person or object detection.
- Water/background labels do not reliably distinguish floodwater from permanent water; shadows, reflections, roofs, and other surfaces can cause errors.
- Binary masks use `uint8` values `0` and `255`. OpenCV-decoded color frames are BGR unless explicitly converted. Preserve source imagery and never silently alter dimensions.
- No machine-learning/deep-learning models, cloud services, web application frameworks, dataset auto-downloads, or person annotations are in scope.
- The current Linux MVP implements these workflows, and the default water baseline has a reproducible 700-pair manual-set agreement result recorded in the roadmap and testing strategy. Windows verification has not been completed. Repository roadmap status is authoritative as work proceeds.

## Brand Commitments

The product name is AquaRescue. The interface must communicate scientific clarity and human review. The user-approved Registered Evidence Folio direction and Evidence Teal palette are documented in `DESIGN.md`; no logo or official affiliation asset has been supplied, so do not fabricate one or imply an affiliation.

## Evidence on Hand

The Floodwater Dataset manual release was evaluated outside the repository; no dataset is stored in the project workspace. A measured result and run-specific processing time are documented, but they are not a claim of general accuracy, floodwater classification, or cross-hardware performance. Candidate proposals have no human-reviewed sanity set or semantic accuracy result. Do not invent or imply any further evidence.

## Product Principles

1. Make every result traceable to its method and parameters.
2. Keep the original input available for comparison and preserve its geometry.
3. Make uncertainty and experimental status visible; keep interpretation with the human.
4. Prefer reproducible, testable image-processing steps over opaque claims.

## Accessibility & Inclusion

Follow `docs/UI_UX_GUIDELINES.md`: keyboard-operable controls, visible focus, readable contrast, useful empty/error states, and layouts that remain usable when resized. The documented WCAG AA contrast values are a design benchmark; platform rendering still requires verification.
