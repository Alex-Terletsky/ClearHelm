from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtWidgets import QSpinBox, QDoubleSpinBox, QComboBox


# ---- Scroll-guarded input widgets ----

class _NoScrollMixin:
    """Ignore scroll-wheel events unless the widget has keyboard focus."""
    def wheelEvent(self, event):
        if self.hasFocus():
            super().wheelEvent(event)
        else:
            event.ignore()


class NoScrollSpinBox(_NoScrollMixin, QSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)


class NoScrollDoubleSpinBox(_NoScrollMixin, QDoubleSpinBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)


class NoScrollComboBox(_NoScrollMixin, QComboBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)


# ---- Qt signal bridge (thread-safe output routing) ----

class SignalBridge(QObject):
    """Emits Qt signals from arbitrary threads so the UI can safely update."""
    text_received = Signal(str, str)   # (model_name, text)
