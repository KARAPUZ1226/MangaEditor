import math
from PySide6.QtCore import Qt, QPointF, QRectF, QSize, Signal
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QPixmap, QImage,
    QPainterPath, QLinearGradient, QFontDatabase, QFontMetrics
)
from PySide6.QtWidgets import (
    QGraphicsRectItem, QGraphicsPixmapItem, QGraphicsTextItem, QGraphicsView, QGraphicsItem,
    QDialog, QVBoxLayout, QListWidget, QListWidgetItem, QHBoxLayout, QPushButton, QMessageBox, QFileDialog,
    QApplication
)

class LayerItem(QGraphicsPixmapItem):
    """Custom graphics item representing a single raster layer (e.g. Clean layer, imported images)."""
    def __init__(self, width, height, name="Слой", parent=None):
        super().__init__(parent)
        self.width = width
        self.height = height
        self.layer_name = name
        self.is_visible = True
        
        # Transparent ARGB image for drawing
        self.image = QImage(width, height, QImage.Format_ARGB32_Premultiplied)
        self.image.fill(Qt.transparent)
        self.setPixmap(QPixmap.fromImage(self.image))
        
        self.scale_factor = 1.0
        self.rotation_angle = 0.0

    def draw_line(self, start, end, tool, color, size, pristine_qimage=None):
        painter = QPainter(self.image)
        painter.setRenderHint(QPainter.Antialiasing)
        if tool == "brush":
            pen = QPen(color, size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(start, end)
        elif tool == "eraser":
            painter.setCompositionMode(QPainter.CompositionMode_Clear)
            pen = QPen(Qt.transparent, size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(start, end)
        elif tool == "inpaint":
            pen = QPen(QColor(239, 68, 68, 160), size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(start, end)
        elif tool == "restore" and pristine_qimage is not None:
            # Создаем текстурную кисть на основе оригинального чистого изображения
            # Это позволяет восстанавливать исходные пиксели точно по координатам рисования
            pen = QPen(QBrush(pristine_qimage), size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(start, end)
        painter.end()
        self.setPixmap(QPixmap.fromImage(self.image))

    def clear(self):
        self.image.fill(Qt.transparent)
        self.setPixmap(QPixmap.fromImage(self.image))


class TypesetTextItem(QGraphicsTextItem):
    """Advanced Typesetting Text Item with outline, soft shadow, glows, gradient, and deformation."""
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsItem.GraphicsItemFlag.ItemIsFocusable
        )
        self.font_color = QColor(0, 0, 0)
        self.outline_color = QColor(255, 255, 255)
        self.outline_width = 3.0
        
        self.has_shadow = False
        self.shadow_color = QColor(0, 0, 0, 150)
        self.shadow_offset = QPointF(3, 3)
        
        self.has_glow = False
        self.glow_color = QColor(6, 182, 212, 180)
        self.glow_radius = 8
        self.glow_type = "soft"
        
        self.use_gradient = False
        self.gradient_color1 = QColor(0, 0, 0)
        self.gradient_color2 = QColor(150, 150, 150)
        
        self.is_vertical = False
        self.deform_type = "None"
        self.raw_text = text
        self.setTextInteractionFlags(Qt.NoTextInteraction)

    def mouseDoubleClickEvent(self, event):
        if self.textInteractionFlags() == Qt.NoTextInteraction:
            if self.is_vertical:
                # Временно показываем горизонтальный текст для удобства редактирования
                super().setPlainText(self.raw_text)
            self.setTextInteractionFlags(Qt.TextEditorInteraction)
            self.setFocus()
        super().mouseDoubleClickEvent(event)

    def focusOutEvent(self, event):
        self.setTextInteractionFlags(Qt.NoTextInteraction)
        self.setSelected(False)
        self.raw_text = self.toPlainText()
        if self.is_vertical:
            self.apply_text_layout()
        super().focusOutEvent(event)

    def setPlainText(self, text):
        self.raw_text = text
        self.apply_text_layout()

    def apply_text_layout(self):
        if self.is_vertical:
            lines = []
            for paragraph in self.raw_text.split('\n'):
                if not paragraph:
                    lines.append("")
                    continue
                lines.extend(list(paragraph))
            super().setPlainText("\n".join(lines))
        else:
            super().setPlainText(self.raw_text)
        self.update()

    def set_vertical(self, vertical):
        self.is_vertical = vertical
        self.apply_text_layout()

    def paint(self, painter, option, widget):
        if self.textInteractionFlags() & Qt.TextEditorInteraction:
            super().paint(painter, option, widget)
            return

        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)
        
        text = self.toPlainText()
        font = self.font()
        painter.setFont(font)
        
        metrics = painter.fontMetrics()
        lines = text.split('\n')
        
        y_offset = metrics.ascent()
        
        # Получаем параметры выравнивания и ширину текстового блока
        align = self.document().defaultTextOption().alignment()
        text_width = self.textWidth()
        if text_width <= 0:
            text_width = self.boundingRect().width()

        # Draw shadow
        if self.has_shadow:
            painter.save()
            painter.translate(self.shadow_offset)
            shadow_y = y_offset
            for line in lines:
                path = QPainterPath()
                line_width = metrics.horizontalAdvance(line)
                if align & Qt.AlignHCenter:
                    x = (text_width - line_width) / 2.0
                elif align & Qt.AlignRight:
                    x = text_width - line_width
                else:
                    x = 0.0
                path.addText(x, shadow_y, font, line)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(self.shadow_color))
                painter.drawPath(path)
                shadow_y += metrics.lineSpacing()
            painter.restore()

        # Draw glow
        if self.has_glow:
            painter.save()
            glow_y = y_offset
            for line in lines:
                path = QPainterPath()
                line_width = metrics.horizontalAdvance(line)
                if align & Qt.AlignHCenter:
                    x = (text_width - line_width) / 2.0
                elif align & Qt.AlignRight:
                    x = text_width - line_width
                else:
                    x = 0.0
                path.addText(x, glow_y, font, line)
                loops = self.glow_radius if self.glow_type == "soft" else 2
                for i in range(loops, 0, -1):
                    alpha = int((180 / loops) * (loops - i + 1))
                    col = QColor(self.glow_color)
                    col.setAlpha(alpha)
                    pen = QPen(col, self.outline_width + i * 2)
                    pen.setJoinStyle(Qt.RoundJoin)
                    painter.setPen(pen)
                    painter.drawPath(path)
                glow_y += metrics.lineSpacing()
            painter.restore()

        # Draw outline
        if self.outline_width > 0:
            outline_y = y_offset
            for line in lines:
                path = QPainterPath()
                line_width = metrics.horizontalAdvance(line)
                if align & Qt.AlignHCenter:
                    x = (text_width - line_width) / 2.0
                elif align & Qt.AlignRight:
                    x = text_width - line_width
                else:
                    x = 0.0
                path.addText(x, outline_y, font, line)
                pen = QPen(self.outline_color, self.outline_width)
                pen.setJoinStyle(Qt.RoundJoin)
                painter.setPen(pen)
                painter.drawPath(path)
                outline_y += metrics.lineSpacing()

        # Draw text fill
        fill_y = y_offset
        for line in lines:
            path = QPainterPath()
            line_width = metrics.horizontalAdvance(line)
            if align & Qt.AlignHCenter:
                x = (text_width - line_width) / 2.0
            elif align & Qt.AlignRight:
                x = text_width - line_width
            else:
                x = 0.0
            path.addText(x, fill_y, font, line)
            if self.use_gradient:
                grad = QLinearGradient(x, fill_y - metrics.ascent(), x, fill_y)
                grad.setColorAt(0, self.font_color)
                grad.setColorAt(1, self.gradient_color2)
                painter.setBrush(QBrush(grad))
            else:
                painter.setBrush(QBrush(self.font_color))
            painter.setPen(Qt.NoPen)
            painter.drawPath(path)
            fill_y += metrics.lineSpacing()


class InteractiveBubbleItem(QGraphicsRectItem):
    """An interactive speech bubble bounding box that can be resized and moved by the user."""
    def __init__(self, rect, parent=None):
        super().__init__(rect, parent)
        self.setFlags(
            QGraphicsRectItem.ItemIsMovable |
            QGraphicsRectItem.ItemIsSelectable |
            QGraphicsRectItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setPen(QPen(QColor(6, 182, 212), 2, Qt.DashLine))
        self.setBrush(QBrush(QColor(6, 182, 212, 30)))
        
        self.handle_size = 12  # Увеличенный размер уголков для удобства
        self.active_handle = None
        self.is_resizing = False

    def paint(self, painter, option, widget):
        super().paint(painter, option, widget)
        if self.isSelected() or self.isUnderMouse():
            painter.setPen(QPen(QColor(239, 68, 68), 1))
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            rect = self.rect()
            # Draw corner handles
            painter.drawRect(QRectF(rect.topLeft(), QSize(self.handle_size, self.handle_size)))
            painter.drawRect(QRectF(rect.topRight() - QPointF(self.handle_size, 0), QSize(self.handle_size, self.handle_size)))
            painter.drawRect(QRectF(rect.bottomLeft() - QPointF(0, self.handle_size), QSize(self.handle_size, self.handle_size)))
            painter.drawRect(QRectF(rect.bottomRight() - QPointF(self.handle_size, self.handle_size), QSize(self.handle_size, self.handle_size)))

    def hoverMoveEvent(self, event):
        rect = self.rect()
        pos = event.pos()
        margin = self.handle_size + 4
        
        if (pos - rect.topLeft()).manhattanLength() < margin:
            self.setCursor(Qt.SizeFDiagCursor)
        elif (pos - rect.topRight()).manhattanLength() < margin:
            self.setCursor(Qt.SizeBDiagCursor)
        elif (pos - rect.bottomLeft()).manhattanLength() < margin:
            self.setCursor(Qt.SizeBDiagCursor)
        elif (pos - rect.bottomRight()).manhattanLength() < margin:
            self.setCursor(Qt.SizeFDiagCursor)
        else:
            self.setCursor(Qt.SizeAllCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        rect = self.rect()
        pos = event.pos()
        margin = self.handle_size + 4
        
        if (pos - rect.topLeft()).manhattanLength() < margin:
            self.active_handle = "topleft"
            self.is_resizing = True
        elif (pos - rect.topRight()).manhattanLength() < margin:
            self.active_handle = "topright"
            self.is_resizing = True
        elif (pos - rect.bottomLeft()).manhattanLength() < margin:
            self.active_handle = "bottomleft"
            self.is_resizing = True
        elif (pos - rect.bottomRight()).manhattanLength() < margin:
            self.active_handle = "bottomright"
            self.is_resizing = True
        else:
            self.active_handle = None
            self.is_resizing = False
            
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_resizing and self.active_handle:
            rect = self.rect()
            pos = event.pos()
            
            # Предварительно устанавливаем новые координаты, но проверяем минимальный размер
            min_size = 20
            
            if self.active_handle == "topleft":
                if rect.bottom() - pos.y() > min_size and rect.right() - pos.x() > min_size:
                    rect.setTopLeft(pos)
            elif self.active_handle == "topright":
                if rect.bottom() - pos.y() > min_size and pos.x() - rect.left() > min_size:
                    rect.setTopRight(pos)
            elif self.active_handle == "bottomleft":
                if pos.y() - rect.top() > min_size and rect.right() - pos.x() > min_size:
                    rect.setBottomLeft(pos)
            elif self.active_handle == "bottomright":
                if pos.y() - rect.top() > min_size and pos.x() - rect.left() > min_size:
                    rect.setBottomRight(pos)
                    
            self.setRect(rect.normalized())
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.is_resizing = False
        self.active_handle = None
        super().mouseReleaseEvent(event)


class MangaGraphicsView(QGraphicsView):
    """Visual graphics view with zooming, panning, and selection events."""
    drawing_started = Signal(QPointF)
    drawing_moved = Signal(QPointF, QPointF)
    drawing_ended = Signal()
    selection_made = Signal(QRectF, bool)
    cutting_line_changed = Signal(float)

    def __init__(self, scene, parent=None):
        super().__init__(scene, parent)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setDragMode(QGraphicsView.NoDrag)
        
        self.is_panning = False
        self.pan_start_pos = None
        self.tool = "pan"
        
        self.selection_start = None
        self.selection_rect_item = None
        self.cutting_lines = []
        self.first_resize = True

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.first_resize and self.scene() and self.scene().items():
            bg_items = [item for item in self.scene().items() if isinstance(item, QGraphicsPixmapItem)]
            if bg_items:
                self.fitInView(bg_items[-1], Qt.KeepAspectRatio)
                self.first_resize = False

    def set_tool(self, tool):
        self.tool = tool
        if tool == "pan":
            self.setCursor(Qt.OpenHandCursor)
        elif tool in ["brush", "eraser", "inpaint", "restore", "draw_bubble"]:
            self.setCursor(Qt.CrossCursor)
        elif tool == "select":
            self.setCursor(Qt.SizeAllCursor)
        elif tool == "cut":
            self.setCursor(Qt.SplitVCursor)
        else:
            self.setCursor(Qt.ArrowCursor)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            if self.scene():
                for item in self.scene().selectedItems():
                    if isinstance(item, InteractiveBubbleItem) or isinstance(item, TypesetTextItem):
                        self.scene().removeItem(item)
            event.accept()
        else:
            super().keyPressEvent(event)

    def draw_cutting_lines(self):
        for item in list(self.scene().items()):
            if isinstance(item, QGraphicsRectItem) and getattr(item, "is_cut_line", False):
                self.scene().removeItem(item)
                
        for y in self.cutting_lines:
            line_item = QGraphicsRectItem(0, y, self.scene().width(), 4)
            line_item.is_cut_line = True
            line_item.setBrush(QBrush(QColor(239, 68, 68)))
            line_item.setPen(Qt.NoPen)
            self.scene().addItem(line_item)

    def wheelEvent(self, event):
        zoom_factor = 1.15
        if event.angleDelta().y() < 0:
            zoom_factor = 1.0 / zoom_factor
        self.scale(zoom_factor, zoom_factor)

    def mousePressEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())
        
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and self.tool == "pan"):
            self.is_panning = True
            self.pan_start_pos = event.position().toPoint()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
            
        if event.button() == Qt.LeftButton:
            if self.tool in ["brush", "eraser", "inpaint", "restore"]:
                self.drawing_started.emit(scene_pos)
                self.last_scene_pos = scene_pos
                event.accept()
                return
            elif self.tool in ["select", "draw_bubble"]:
                self.selection_start = scene_pos
                self.selection_rect_item = QGraphicsRectItem()
                col = QColor(6, 182, 212) if self.tool == "select" else QColor(168, 85, 247)
                self.selection_rect_item.setPen(QPen(col, 2, Qt.DashLine))
                self.selection_rect_item.setBrush(QBrush(QColor(col.red(), col.green(), col.blue(), 40)))
                self.scene().addItem(self.selection_rect_item)
                event.accept()
                return
            elif self.tool == "cut":
                y_val = scene_pos.y()
                self.cutting_lines.append(y_val)
                self.draw_cutting_lines()
                self.cutting_line_changed.emit(y_val)
                event.accept()
                return
                
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        scene_pos = self.mapToScene(event.position().toPoint())
        
        if self.is_panning:
            delta = event.position().toPoint() - self.pan_start_pos
            h_bar = self.horizontalScrollBar()
            v_bar = self.verticalScrollBar()
            h_bar.setValue(h_bar.value() - delta.x())
            v_bar.setValue(v_bar.value() - delta.y())
            self.pan_start_pos = event.position().toPoint()
            event.accept()
            return
            
        if event.buttons() & Qt.LeftButton:
            if self.tool in ["brush", "eraser", "inpaint", "restore"]:
                self.drawing_moved.emit(self.last_scene_pos, scene_pos)
                self.last_scene_pos = scene_pos
                event.accept()
                return
            elif self.tool in ["select", "draw_bubble"] and self.selection_start:
                rect = QRectF(self.selection_start, scene_pos).normalized()
                self.selection_rect_item.setRect(rect)
                event.accept()
                return
                
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MiddleButton or (event.button() == Qt.LeftButton and self.is_panning):
            self.is_panning = False
            self.set_tool(self.tool)
            event.accept()
            return
            
        if event.button() == Qt.LeftButton:
            if self.tool in ["brush", "eraser", "inpaint", "restore"]:
                self.drawing_ended.emit()
                event.accept()
                return
            elif self.tool in ["select", "draw_bubble"] and self.selection_rect_item:
                rect = self.selection_rect_item.rect()
                self.scene().removeItem(self.selection_rect_item)
                self.selection_rect_item = None
                self.selection_start = None
                
                # Check for shift modifier
                is_shift_pressed = bool(QApplication.keyboardModifiers() & Qt.ShiftModifier)
                self.selection_made.emit(rect, is_shift_pressed)
                event.accept()
                return
                
        super().mouseReleaseEvent(event)


class FontManagerDialog(QDialog):
    def __init__(self, font_list, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Менеджер шрифтов")
        self.resize(400, 300)
        self.font_list = font_list
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        self.list_widget = QListWidget()
        for font in self.font_list:
            self.list_widget.addItem(font)
        layout.addWidget(self.list_widget)
        
        btn_layout = QHBoxLayout()
        self.btn_add = QPushButton("Добавить шрифт (.ttf/.otf)")
        self.btn_add.clicked.connect(self.add_font)
        btn_layout.addWidget(self.btn_add)
        
        self.btn_close = QPushButton("Закрыть")
        self.btn_close.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_close)
        layout.addLayout(btn_layout)

    def add_font(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Добавить шрифт", "", "Шрифты (*.ttf *.otf)"
        )
        if file_path:
            font_id = QFontDatabase.addApplicationFont(file_path)
            if font_id != -1:
                families = QFontDatabase.applicationFontFamilies(font_id)
                if families:
                    family = families[0]
                    if family not in self.font_list:
                        self.font_list.append(family)
                        self.list_widget.addItem(family)
                        QMessageBox.information(self, "Успех", f"Шрифт '{family}' успешно добавлен в проект!")
            else:
                QMessageBox.warning(self, "Ошибка", "Не удалось загрузить шрифт.")
