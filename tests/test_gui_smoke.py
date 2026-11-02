"""Headless startup checks for the PySide6 rescue-first desktop workspace."""

from app.window import AquaRescueWindow


def test_main_window_starts_in_rescue_analysis_without_fake_metrics(
    qt_application,
) -> None:
    window = AquaRescueWindow()

    assert window.windowTitle() == "AquaRescue"
    assert window.tabs.count() == 3
    assert window.tabs.tabText(0) == "Rescue Analysis"
    assert window.tabs.tabText(1) == "Water Analysis"
    assert window.tabs.tabText(2) == "Evaluation"
    assert window.tabs.currentIndex() == 0
    assert window.rescue_workspace.advanced_panel.isHidden()
    assert window.rescue_workspace.run_button.text() == "Analyze"
    assert not window.rescue_workspace.run_button.isEnabled()
    assert window.image_workspace.total_pixels_value.text() == "—"
    assert window.evaluation_workspace.iou_value.text() == "—"

    window.resize(1000, 680)
    assert window.width() >= 1000
    assert window.height() >= 680

    window.close()
