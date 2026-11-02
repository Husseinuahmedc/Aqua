"""Shared, explicit save-path selection with overwrite confirmation."""

from pathlib import Path

from PySide6.QtWidgets import QFileDialog, QMessageBox, QWidget


def choose_save_path(
    parent: QWidget,
    *,
    title: str,
    suggested_path: Path,
    file_filter: str,
    default_suffix: str,
) -> tuple[Path, bool] | None:
    """Return a selected path and overwrite consent, or ``None`` on cancel."""
    selected, _ = QFileDialog.getSaveFileName(
        parent,
        title,
        str(suggested_path),
        file_filter,
    )
    if not selected:
        return None
    path = Path(selected)
    if not path.suffix:
        path = path.with_suffix(default_suffix)
    overwrite = False
    if path.exists():
        answer = QMessageBox.question(
            parent,
            "Replace existing file?",
            f"'{path.name}' already exists. Replace it?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return None
        overwrite = True
    return path, overwrite
