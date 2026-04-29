from PySide6.QtCore import Qt, QObject, Signal
from PySide6.QtGui import QFontMetrics, QPixmap
from PySide6.QtWidgets import (
    QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QSizePolicy,
    QWidget, QLabel, QPushButton,
)


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


# ---- Auto-growing plain text input (Enter sends, Shift+Enter newline) ----

class PromptInput(QPlainTextEdit):
    """Single-line-looking text input that supports Shift+Enter for newlines.

    Emits ``submitted`` when the user presses Enter (without Shift).
    Auto-grows up to *max_lines* then scrolls.
    """
    submitted = Signal()

    def __init__(self, max_lines: int = 4, match_widget=None, parent=None):
        super().__init__(parent)
        self._max_lines = max_lines
        self._match_widget = match_widget  # widget whose height sets the 1-line height
        self._target_h = 0
        self._base_h = 0
        self._line_h = 0
        self.setTabChangesFocus(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.document().setDocumentMargin(0)
        self.setStyleSheet("QPlainTextEdit { padding: 9px 6px 1px 6px; }")
        self.textChanged.connect(self._recalc_height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def _recalc_height(self):
        if self._base_h == 0:
            return  # not shown yet — nothing to measure against

        line_count = max(1, self.toPlainText().count('\n') + 1)
        clamped = min(line_count, self._max_lines)

        new_h = self._base_h + (clamped - 1) * self._line_h

        if new_h != self._target_h:
            self._target_h = new_h
            self.setFixedHeight(new_h)

        # Scroll control: only scroll when content exceeds max_lines
        if line_count > self._max_lines:
            self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        else:
            self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            sb = self.verticalScrollBar()
            if sb.value() != 0:
                sb.setValue(0)

    def showEvent(self, event):
        super().showEvent(event)
        if self._base_h == 0:
            self._line_h = QFontMetrics(self.font()).lineSpacing()
            if self._match_widget:
                self._base_h = self._match_widget.sizeHint().height()
            else:
                vm = self.viewportMargins()
                fw = self.frameWidth()
                self._base_h = int(self._line_h + vm.top() + vm.bottom()
                                   + fw * 2)
            self._recalc_height()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._recalc_height()

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if event.modifiers() & Qt.ShiftModifier:
                super().keyPressEvent(event)
                self._recalc_height()
            else:
                self.submitted.emit()
            return
        super().keyPressEvent(event)

    def text(self) -> str:
        """Convenience: match QLineEdit.text() API."""
        return self.toPlainText()


# ---- Image attachment thumbnail ----

class ImageThumbnail(QWidget):
    """Attachment thumbnail with an overlaid delete button and click-to-expand.

    Scales the image to 25% of its original size, capped at MAX_SIZE on the
    longer side. Aspect ratio is preserved and the widget shrinks to match.
    """
    MIN_SIZE = 64
    MAX_SIZE = 160
    SCALE = 0.25

    clicked = Signal(str)
    removed = Signal(str)

    def __init__(self, image_path: str, parent=None):
        super().__init__(parent)
        self._path = image_path

        pix = QPixmap(image_path)
        if not pix.isNull():
            longer = max(pix.width(), pix.height())
            # Target longer side = 25% of source, clamped to [MIN, MAX]
            target_longer = min(self.MAX_SIZE,
                                max(self.MIN_SIZE, round(longer * self.SCALE)))
            scale = target_longer / longer
            tw = max(1, round(pix.width() * scale))
            th = max(1, round(pix.height() * scale))
            scaled = pix.scaled(tw, th, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            tw, th = self.MIN_SIZE, self.MIN_SIZE
            scaled = None

        self.setFixedSize(tw, th)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip(f"{image_path}\n(click to expand)")

        self._label = QLabel(self)
        self._label.setGeometry(0, 0, tw, th)
        self._label.setAlignment(Qt.AlignCenter)
        self._label.setStyleSheet(
            "QLabel { background: #1e1e2e; border: 1px solid #45475a; border-radius: 4px; }"
        )
        if scaled is not None:
            self._label.setPixmap(scaled)
        else:
            self._label.setText("?")

        btn_size = 20
        inset = 4
        self._del_btn = QPushButton("\u2715", self)
        self._del_btn.setFixedSize(btn_size, btn_size)
        self._del_btn.move(tw - btn_size - inset, inset)
        self._del_btn.setCursor(Qt.PointingHandCursor)
        self._del_btn.setToolTip("Remove attachment")
        self._del_btn.setStyleSheet(
            "QPushButton { background: #f38ba8; color: #1e1e2e;"
            " border: 1px solid #1e1e2e; border-radius: 0;"
            " font-size: 14px; font-weight: bold; padding: 0; }"
            "QPushButton:hover { background: #eba0ac; }"
        )
        self._del_btn.clicked.connect(lambda: self.removed.emit(self._path))
        self._del_btn.raise_()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Suppress the expand click when the press lands on the × button
            if self.childAt(event.pos()) is not self._del_btn:
                self.clicked.emit(self._path)
        super().mousePressEvent(event)


# ---- Qt signal bridge (thread-safe output routing) ----

class SignalBridge(QObject):
    """Emits Qt signals from arbitrary threads so the UI can safely update."""
    text_received = Signal(str, str)        # (model_name, text)
    generation_complete = Signal(str, str)  # (model_name, full_response)
