"""Shared restrained theme tokens for the scientific desktop workspace."""

from PySide6.QtWidgets import QApplication


COLORS = {
    "canvas": "#F2F5F2",
    "surface": "#FFFFFF",
    "side_surface": "#E7EEEA",
    "ink": "#142B2A",
    "muted": "#415852",
    "border": "#C9D6D0",
    "accent": "#137A70",
    "accent_hover": "#0F5D56",
    "accent_soft": "#DCEBE6",
    "caution": "#87521B",
    "caution_surface": "#F6EFE3",
    "error": "#963B35",
    "success": "#226B49",
    "disabled_surface": "#E8ECE9",
    "pressed_surface": "#C8DED6",
    "header": "#142B2A",
    "header_text": "#F5F8F5",
    "header_secondary": "#D2E0DB",
}


def apply_application_theme(application: QApplication) -> None:
    """Apply consistent colors and native Fusion controls to the application."""
    application.setStyle("Fusion")
    application.setStyleSheet(
        f"""
        QWidget {{
            background: {COLORS['canvas']};
            color: {COLORS['ink']};
            font-size: 13px;
        }}
        QWidget#applicationHeader {{
            background: {COLORS['header']};
            color: {COLORS['header_text']};
        }}
        QLabel#applicationTitle {{
            background: transparent;
            color: {COLORS['header_text']};
            font-size: 20px;
            font-weight: 650;
        }}
        QLabel#headerSecondary {{
            background: transparent;
            color: {COLORS['header_secondary']};
        }}
        QLabel {{ background: transparent; }}
        QLabel#mutedText {{ color: {COLORS['muted']}; }}
        QLabel#metricValue {{ font-weight: 650; }}
        QLabel#registrationStrip {{
            background: {COLORS['accent_soft']};
            border: 1px solid {COLORS['border']};
            padding: 8px 12px;
        }}
        QLabel#resultState {{
            background: {COLORS['side_surface']};
            border: 1px solid {COLORS['border']};
            padding: 6px 10px;
        }}
        QLabel#resultState[stale="true"] {{
            background: {COLORS['caution_surface']};
            color: {COLORS['caution']};
            font-weight: 650;
        }}
        QTabWidget::pane {{
            border: 1px solid {COLORS['border']};
            background: {COLORS['canvas']};
        }}
        QTabBar::tab {{
            background: {COLORS['side_surface']};
            color: {COLORS['muted']};
            padding: 8px 12px;
            margin-right: 3px;
            border: 1px solid {COLORS['border']};
            border-bottom: none;
        }}
        QTabBar::tab:selected {{
            background: {COLORS['accent_soft']};
            color: {COLORS['ink']};
            font-weight: 650;
            border-bottom: 2px solid {COLORS['accent']};
        }}
        QTabBar::tab:focus {{ outline: 2px solid {COLORS['accent']}; }}
        QGroupBox {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 6px;
            margin-top: 12px;
            padding: 7px;
            font-weight: 600;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 8px;
            padding: 0 5px;
        }}
        QPushButton {{
            background: {COLORS['surface']};
            color: {COLORS['ink']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            padding: 8px 12px;
            min-height: 20px;
        }}
        QPushButton:hover {{ background: {COLORS['accent_soft']}; }}
        QPushButton:pressed {{ background: {COLORS['pressed_surface']}; }}
        QPushButton:focus {{ border: 2px solid {COLORS['accent']}; }}
        QPushButton:disabled {{
            color: {COLORS['muted']};
            background: {COLORS['disabled_surface']};
        }}
        QPushButton#primaryAction {{
            background: {COLORS['accent']};
            color: {COLORS['surface']};
            border-color: {COLORS['accent']};
            font-weight: 650;
        }}
        QPushButton#primaryAction:hover {{ background: {COLORS['accent_hover']}; }}
        QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {{
            background: {COLORS['surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            padding: 6px 8px;
            min-height: 20px;
        }}
        QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
        QLineEdit:focus {{ border: 2px solid {COLORS['accent']}; }}
        QSlider::groove:horizontal {{
            height: 5px;
            background: {COLORS['border']};
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            background: {COLORS['accent']};
            width: 14px;
            margin: -5px 0;
            border-radius: 7px;
        }}
        QSlider:focus {{ outline: 2px solid {COLORS['accent']}; }}
        QProgressBar {{
            background: {COLORS['side_surface']};
            border: 1px solid {COLORS['border']};
            border-radius: 4px;
            text-align: center;
            min-height: 16px;
        }}
        QProgressBar::chunk {{ background: {COLORS['accent']}; border-radius: 3px; }}
        QStatusBar {{ background: {COLORS['side_surface']}; color: {COLORS['ink']}; }}
        QToolTip {{
            background: {COLORS['ink']};
            color: {COLORS['surface']};
            border: 1px solid {COLORS['ink']};
            padding: 5px;
        }}
        QScrollArea {{ border: none; background: transparent; }}
        QScrollArea > QWidget > QWidget {{ background: transparent; }}
        """
    )
