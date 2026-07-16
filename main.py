import sys
import os
import zipfile
import json
import math
import cv2
import numpy as np
import onnxruntime as ort

from PySide6.QtCore import Qt, QPointF, QRectF, QSize
from PySide6.QtGui import (
    QAction, QPainter, QPen, QBrush, QColor, QFont, QPixmap, QImage,
    QUndoStack, QUndoCommand, QKeySequence, QShortcut, QFontDatabase, QFontMetrics
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLabel, QFileDialog, QListWidget, QComboBox, QSpinBox, QListWidgetItem,
    QDoubleSpinBox, QGraphicsScene, QGraphicsRectItem, QGroupBox, QSlider, QTabWidget,
    QTextEdit, QMessageBox, QColorDialog, QSplitter, QCheckBox, QDialog, QProgressDialog
)
from core_gui import LayerItem, TypesetTextItem, InteractiveBubbleItem, MangaGraphicsView, FontManagerDialog
from cleaner import smart_clean_bubbles, smart_inpaint_rect, LaMaInpainter
from translator import EasyOCRManager, translate_google, EASYOCR_AVAILABLE
from waifu import WaifuDialog

try:
    from psd_tools import PSDImage
    PSD_TOOLS_AVAILABLE = True
except ImportError:
    PSD_TOOLS_AVAILABLE = False

try:
    import pytoshop
    import pytoshop.user.nested_layers
    PYTOSHOP_AVAILABLE = True
except ImportError:
    PYTOSHOP_AVAILABLE = False

# Unicode-safe image IO functions to fix OpenCV Windows bugs with Cyrillic paths
def cv_imread_unicode(file_path):
    try:
        img_array = np.fromfile(file_path, dtype=np.uint8)
        return cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"Unicode load error: {e}")
        return None

def cv_imwrite_unicode(file_path, img):
    try:
        ext = os.path.splitext(file_path)[1]
        if not ext:
            ext = ".png"
        is_success, im_buf_arr = cv2.imencode(ext, img)
        if is_success:
            im_buf_arr.tofile(file_path)
            return True
    except Exception as e:
        print(f"Unicode save error: {e}")
    return False

# Dark Theme CSS Style
DARK_THEME = """
QMainWindow {
    background-color: #121214;
    color: #e3e3e6;
}
QWidget {
    background-color: #121214;
    color: #e3e3e6;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #27272a;
    border-radius: 8px;
    margin-top: 16px;
    font-weight: bold;
    padding-top: 12px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #06b6d4;
}
QPushButton {
    background-color: #1e1e22;
    border: 1px solid #27272a;
    border-radius: 6px;
    padding: 6px 14px;
    min-height: 22px;
    color: #f4f4f5;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #06b6d4;
    color: #121214;
    border-color: #0891b2;
}
QPushButton:pressed {
    background-color: #0891b2;
}
QPushButton:checked {
    background-color: #06b6d4;
    color: #121214;
    border-color: #0891b2;
}
QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QLineEdit {
    background-color: #18181b;
    border: 1px solid #27272a;
    border-radius: 6px;
    padding: 6px;
    color: #f4f4f5;
}
QComboBox:hover, QSpinBox:hover, QTextEdit:hover {
    border-color: #3f3f46;
}
QComboBox::drop-down {
    border: 0px;
}
QSlider::groove:horizontal {
    border: 1px solid #27272a;
    height: 6px;
    background: #18181b;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #06b6d4;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background: #22d3ee;
}
QListWidget {
    background-color: #18181b;
    border: 1px solid #27272a;
    border-radius: 6px;
    color: #f4f4f5;
    padding: 4px;
}
QListWidget::item {
    padding: 6px;
    border-radius: 4px;
}
QListWidget::item:selected {
    background-color: #06b6d4;
    color: #121214;
    font-weight: bold;
}
QGraphicsView {
    border: 1px solid #27272a;
    background-color: #09090b;
    border-radius: 8px;
}
QTabBar::tab {
    background: #18181b;
    border: 1px solid #27272a;
    border-bottom: none;
    padding: 8px 16px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    color: #a1a1aa;
}
QTabBar::tab:selected {
    background: #27272a;
    color: #06b6d4;
    border-bottom: 2px solid #06b6d4;
}
QSplitter::handle {
    background-color: #27272a;
}
"""

# --- UNDO / REDO COMMANDS ---
class UndoPaintCommand(QUndoCommand):
    def __init__(self, layer_item, before_image, after_image, description="Рисование", parent_editor=None):
        super().__init__(description)
        self.layer_item = layer_item
        self.before_image = before_image.copy()
        self.after_image = after_image.copy()
        self.parent_editor = parent_editor
        self.is_bg_layer = (parent_editor is not None and parent_editor.layers and layer_item == parent_editor.layers[0])
        self.before_cv_image = None
        self.after_cv_image = None
        
    def set_cv_images(self, before_cv, after_cv):
        if self.is_bg_layer:
            self.before_cv_image = before_cv.copy() if before_cv is not None else None
            self.after_cv_image = after_cv.copy() if after_cv is not None else None

    def undo(self):
        self.layer_item.image = self.before_image.copy()
        self.layer_item.setPixmap(QPixmap.fromImage(self.layer_item.image))
        if self.is_bg_layer and self.parent_editor and self.before_cv_image is not None:
            self.parent_editor.original_cv_image = self.before_cv_image.copy()
        if self.parent_editor:
            self.parent_editor.sync_history_list()

    def redo(self):
        self.layer_item.image = self.after_image.copy()
        self.layer_item.setPixmap(QPixmap.fromImage(self.layer_item.image))
        if self.is_bg_layer and self.parent_editor and self.after_cv_image is not None:
            self.parent_editor.original_cv_image = self.after_cv_image.copy()
        if self.parent_editor:
            self.parent_editor.sync_history_list()


class UndoAddTextCommand(QUndoCommand):
    def __init__(self, scene, text_item, parent_editor=None):
        super().__init__("Добавление текста")
        self.scene = scene
        self.text_item = text_item
        self.parent_editor = parent_editor

    def undo(self):
        self.scene.removeItem(self.text_item)
        if self.parent_editor:
            self.parent_editor.sync_history_list()

    def redo(self):
        self.scene.addItem(self.text_item)
        if self.parent_editor:
            self.parent_editor.sync_history_list()


class MangaEditorApp(QMainWindow):
    """Main Application Window."""
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Antigravity Manga/Manhwa Editor Pro")
        self.resize(1300, 900)
        self.setMinimumSize(QSize(700, 500))
        self.setStyleSheet(DARK_THEME)
        self.setAcceptDrops(True)
        # State Data
        self.project_path = None
        self.original_cv_image = None
        self.pristine_cv_image = None
        self.pristine_qimage = None
        self.lama_inpainter = None

        self.layers = []
        self.project_fonts = ["Segoe UI", "Arial", "Impact", "Comic Sans MS"]
        self.current_text_item = None
        
        # Color states
        self.text_color = QColor(0, 0, 0)
        self.outline_color = QColor(255, 255, 255)
        self.gradient_color2 = QColor(150, 150, 150)
        self.brush_color = QColor(255, 255, 255)
        
        # Lazy OCR Manager
        self.ocr_manager = EasyOCRManager()
        
        # Selection area rectangle coordinates
        self.selection_area_rect = None
        self.selection_visual_item = None
        
        # Default cleaner layer opacity setting (0-100)
        self.default_clean_opacity = 100
        
        # Drawing Undo backup
        self.draw_start_image = None
        
        # Undo/Redo Stack
        self.undo_stack = QUndoStack(self)
        
        # Directory Navigation
        self.folder_images = []
        self.current_folder_index = -1
        
        self.scene = QGraphicsScene(self)
        self.view = MangaGraphicsView(self.scene, self)
        
        self.setup_ui()
        self.setup_canvas_signals()
        self.setup_shortcuts()
        
        # Инициализация U-Net сегментатора
        self.text_segmenter = None
        self.load_text_segmenter()

    def setup_ui(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("Файл")
        
        act_open = QAction("Открыть файл (Изображение / Проект)", self)
        act_open.triggered.connect(self.load_image_or_project)
        file_menu.addAction(act_open)
        
        act_open_dir = QAction("Открыть папку главы", self)
        act_open_dir.triggered.connect(self.open_folder_dialog)
        file_menu.addAction(act_open_dir)
        
        act_import_psd = QAction("Импорт PSD", self)
        act_import_psd.triggered.connect(self.import_psd)
        file_menu.addAction(act_import_psd)
        
        act_export_psd = QAction("Экспорт PSD", self)
        act_export_psd.triggered.connect(self.export_psd)
        file_menu.addAction(act_export_psd)
        
        act_save_mproj = QAction("Сохранить проект (.mproj)", self)
        act_save_mproj.triggered.connect(self.save_mproj)
        file_menu.addAction(act_save_mproj)
        
        act_save_img = QAction("Экспорт в PNG/JPG", self)
        act_save_img.triggered.connect(self.export_image)
        file_menu.addAction(act_save_img)
        
        main_splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(main_splitter)
        
        # 1. Left Sidebar: Vertical Toolbar
        self.left_toolbar = QWidget()
        self.left_toolbar.setFixedWidth(55)
        self.left_toolbar.setStyleSheet("background-color: #17171a; border-right: 1px solid #27272a;")
        ly_left = QVBoxLayout(self.left_toolbar)
        ly_left.setContentsMargins(4, 10, 4, 10)
        ly_left.setSpacing(10)
        
        self.btn_pan = QPushButton("🔍")
        self.btn_pan.setToolTip("Перемещение (Pan)")
        self.btn_pan.setCheckable(True)
        self.btn_pan.setChecked(True)
        self.btn_pan.clicked.connect(lambda: self.select_tool("pan"))
        ly_left.addWidget(self.btn_pan)
        
        self.btn_brush = QPushButton("🖌")
        self.btn_brush.setToolTip("Кисть (Клин)")
        self.btn_brush.setCheckable(True)
        self.btn_brush.clicked.connect(lambda: self.select_tool("brush"))
        ly_left.addWidget(self.btn_brush)
        
        self.btn_eraser = QPushButton("🧽")
        self.btn_eraser.setToolTip("Ластик")
        self.btn_eraser.setCheckable(True)
        self.btn_eraser.clicked.connect(lambda: self.select_tool("eraser"))
        ly_left.addWidget(self.btn_eraser)
        self.btn_inpaint = QPushButton("🪄")
        self.btn_inpaint.setToolTip("ИИ-Ластик (Inpaint)")
        self.btn_inpaint.setCheckable(True)
        self.btn_inpaint.clicked.connect(lambda: self.select_tool("inpaint"))
        ly_left.addWidget(self.btn_inpaint)

        self.btn_restore = QPushButton("🔄")
        self.btn_restore.setToolTip("Восстанавливающая кисть (Восстановление оригинала)")
        self.btn_restore.setCheckable(True)
        self.btn_restore.clicked.connect(lambda: self.select_tool("restore"))
        ly_left.addWidget(self.btn_restore)
        self.btn_select = QPushButton("🎯")
        self.btn_select.setToolTip("Инструмент выделения (Рамка)")
        self.btn_select.setCheckable(True)
        self.btn_select.clicked.connect(lambda: self.select_tool("select"))
        ly_left.addWidget(self.btn_select)
        
        self.btn_add_bubble_tool = QPushButton("💬")
        self.btn_add_bubble_tool.setToolTip("Инструмент: Рисовать Бабл вручную")
        self.btn_add_bubble_tool.setCheckable(True)
        self.btn_add_bubble_tool.clicked.connect(lambda: self.select_tool("draw_bubble"))
        ly_left.addWidget(self.btn_add_bubble_tool)

        self.btn_text = QPushButton("Ｔ")
        self.btn_text.setToolTip("Добавить Текст (Тайп)")
        self.btn_text.clicked.connect(lambda: self.add_text_item())
        ly_left.addWidget(self.btn_text)
        
        ly_left.addStretch()
        main_splitter.addWidget(self.left_toolbar)
        
        # 2. Central Area: Workspace Canvas
        workspace = QWidget()
        work_layout = QVBoxLayout(workspace)
        work_layout.setContentsMargins(0, 0, 0, 0)
        
        # Navigation control strip
        top_strip = QHBoxLayout()
        top_strip.setContentsMargins(6, 6, 6, 6)
        
        self.btn_prev_page = QPushButton("◀ Пред")
        self.btn_prev_page.clicked.connect(self.prev_page)
        self.btn_prev_page.setEnabled(False)
        top_strip.addWidget(self.btn_prev_page)
        
        self.lbl_page_info = QLabel("Страниц: 0")
        self.lbl_page_info.setStyleSheet("color: #a1a1aa; font-weight: bold; padding: 0 4px;")
        top_strip.addWidget(self.lbl_page_info)
        
        self.btn_next_page = QPushButton("След ▶")
        self.btn_next_page.clicked.connect(self.next_page)
        self.btn_next_page.setEnabled(False)
        top_strip.addWidget(self.btn_next_page)
        
        top_strip.addStretch()
        
        # Selection operations buttons (Clear, Inpaint)
        self.btn_clear_sel = QPushButton("Очистить выделение")
        self.btn_clear_sel.clicked.connect(self.clear_selection_area)
        self.btn_clear_sel.setToolTip("Очистить текущую выделенную область на активном слое")
        top_strip.addWidget(self.btn_clear_sel)
        
        self.btn_inpaint_sel = QPushButton("ИИ-Клин выделения")
        self.btn_inpaint_sel.clicked.connect(self.inpaint_selection_area)
        self.btn_inpaint_sel.setToolTip("Выполнить умный клининг выделенной области")
        top_strip.addWidget(self.btn_inpaint_sel)
        
        top_strip.addSpacing(10)
        
        self.combo_app_mode = QComboBox()
        self.combo_app_mode.addItems(["Режим: Манга (Ч/Б)", "Режим: Манхва (Склейка/Нарезка)"])
        self.combo_app_mode.currentIndexChanged.connect(self.toggle_mode)
        top_strip.addWidget(self.combo_app_mode)
        
        self.chk_grayscale = QCheckBox("Ч/Б Фильтр")
        self.chk_grayscale.clicked.connect(self.toggle_grayscale)
        top_strip.addWidget(self.chk_grayscale)
        self.btn_waifu = QPushButton("Вайфу (Чистка/Апскейл)")
        self.btn_waifu.clicked.connect(self.open_waifu_dialog)
        top_strip.addWidget(self.btn_waifu)
        
        work_layout.addLayout(top_strip)
        work_layout.addWidget(self.view)
        
        self.manhwa_bar = QHBoxLayout()
        self.btn_stitch = QPushButton("Склеить страницы")
        self.btn_stitch.clicked.connect(self.stitch_manhwa_pages)
        self.manhwa_bar.addWidget(self.btn_stitch)
        
        self.btn_cut_mode = QPushButton("Режим нарезки")
        self.btn_cut_mode.setCheckable(True)
        self.btn_cut_mode.clicked.connect(self.toggle_cut_mode)
        self.manhwa_bar.addWidget(self.btn_cut_mode)
        
        self.btn_slice = QPushButton("Нарезать и экспортировать")
        self.btn_slice.clicked.connect(self.slice_manhwa_pages)
        self.manhwa_bar.addWidget(self.btn_slice)
        
        self.manhwa_bar_widget = QWidget()
        self.manhwa_bar_widget.setLayout(self.manhwa_bar)
        self.manhwa_bar_widget.setVisible(False)
        work_layout.addWidget(self.manhwa_bar_widget)
        
        main_splitter.addWidget(workspace)
        
        # 3. Right Sidebar: Controls
        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        
        # Layer Management Widget
        grp_layers = QGroupBox("Слои")
        ly_layers = QVBoxLayout(grp_layers)
        self.list_layers = QListWidget()
        self.list_layers.itemClicked.connect(self.on_layer_selected)
        ly_layers.addWidget(self.list_layers)
        
        # Layer opacity & visibility controls
        ly_layer_ctrl = QHBoxLayout()
        self.chk_layer_visible = QCheckBox("Показ")
        self.chk_layer_visible.setChecked(True)
        self.chk_layer_visible.clicked.connect(self.toggle_active_layer_visibility)
        ly_layer_ctrl.addWidget(self.chk_layer_visible)
        
        ly_layer_ctrl.addWidget(QLabel("Прозрачность:"))
        self.slider_layer_opacity = QSlider(Qt.Horizontal)
        self.slider_layer_opacity.setRange(0, 100)
        self.slider_layer_opacity.setValue(100)
        self.slider_layer_opacity.valueChanged.connect(self.set_active_layer_opacity)
        ly_layer_ctrl.addWidget(self.slider_layer_opacity)
        ly_layers.addLayout(ly_layer_ctrl)
        
        btn_layer_ly = QHBoxLayout()
        self.btn_add_layer = QPushButton("+ Слой")
        self.btn_add_layer.clicked.connect(self.add_blank_layer)
        btn_layer_ly.addWidget(self.btn_add_layer)
        
        self.btn_add_img_layer = QPushButton("+ Картинка")
        self.btn_add_img_layer.clicked.connect(self.add_image_layer)
        btn_layer_ly.addWidget(self.btn_add_img_layer)
        ly_layers.addLayout(btn_layer_ly)
        sidebar_layout.addWidget(grp_layers)
        
        self.tabs = QTabWidget()
        sidebar_layout.addWidget(self.tabs)
        
        # Tab 1: Clean/Klin
        tab_brush = QWidget()
        brush_ly = QVBoxLayout(tab_brush)
        
        brush_grid = QGridLayout()
        brush_grid.addWidget(QLabel("Размер кисти:"), 0, 0)
        self.spin_brush_size = QSpinBox()
        self.spin_brush_size.setRange(1, 150)
        self.spin_brush_size.setValue(15)
        brush_grid.addWidget(self.spin_brush_size, 0, 1)
        
        self.btn_brush_color = QPushButton("Цвет очистки")
        self.btn_brush_color.clicked.connect(self.choose_brush_color)
        self.btn_brush_color.setStyleSheet(f"background-color: {self.brush_color.name()};")
        brush_grid.addWidget(self.btn_brush_color, 1, 0, 1, 2)
        
        # Speech bubble detector operations
        brush_grid.addWidget(QLabel("Управление баблами:"), 2, 0, 1, 2)
        
        self.btn_detect_bubbles = QPushButton("Найти баблы (ИИ)")
        self.btn_detect_bubbles.clicked.connect(self.detect_manga_bubbles)
        brush_grid.addWidget(self.btn_detect_bubbles, 3, 0)
        
        self.btn_delete_bubble = QPushButton("Удалить бабл")
        self.btn_delete_bubble.clicked.connect(self.delete_selected_bubble)
        brush_grid.addWidget(self.btn_delete_bubble, 3, 1)
        
        self.btn_clear_bubbles = QPushButton("Очистить все баблы (Клин)")
        self.btn_clear_bubbles.clicked.connect(self.clear_all_detected_bubbles)
        self.btn_clear_bubbles.setToolTip("Стереть весь текст внутри баблов с сохранением обводки облачков")
        brush_grid.addWidget(self.btn_clear_bubbles, 4, 0, 1, 2)
        
        brush_ly.addLayout(brush_grid)
        brush_ly.addStretch()
        self.tabs.addTab(tab_brush, "Клин")
        
        # Tab 2: Typeset/Font configuration
        tab_type = QWidget()
        type_ly = QVBoxLayout(tab_type)
        self.btn_manage_fonts = QPushButton("Настройка шрифтов проекта")
        self.btn_manage_fonts.clicked.connect(self.open_font_manager)
        type_ly.addWidget(self.btn_manage_fonts)
        
        type_grid = QGridLayout()
        type_grid.addWidget(QLabel("Шрифт:"), 0, 0)
        self.combo_font = QComboBox()
        self.combo_font.addItems(self.project_fonts)
        self.combo_font.currentIndexChanged.connect(self.update_text_properties)
        type_grid.addWidget(self.combo_font, 0, 1)
        
        type_grid.addWidget(QLabel("Размер:"), 1, 0)
        self.spin_font_size = QSpinBox()
        self.spin_font_size.setRange(4, 200)
        self.spin_font_size.setValue(22)
        self.spin_font_size.valueChanged.connect(self.update_text_properties)
        type_grid.addWidget(self.spin_font_size, 1, 1)
        
        self.btn_txt_color = QPushButton("Цвет заливки")
        self.btn_txt_color.clicked.connect(self.choose_text_color)
        self.btn_txt_color.setStyleSheet(f"background-color: {self.text_color.name()};")
        type_grid.addWidget(self.btn_txt_color, 2, 0)
        
        self.btn_outline_color = QPushButton("Цвет контура")
        self.btn_outline_color.clicked.connect(self.choose_outline_color)
        self.btn_outline_color.setStyleSheet(f"background-color: {self.outline_color.name()};")
        type_grid.addWidget(self.btn_outline_color, 2, 1)
        
        type_grid.addWidget(QLabel("Ширина контура:"), 3, 0)
        self.spin_outline = QDoubleSpinBox()
        self.spin_outline.setRange(0.0, 30.0)
        self.spin_outline.setValue(3.0)
        self.spin_outline.valueChanged.connect(self.update_text_properties)
        type_grid.addWidget(self.spin_outline, 3, 1)
        
        self.chk_shadow = QCheckBox("Тень")
        self.chk_shadow.clicked.connect(self.update_text_properties)
        type_grid.addWidget(self.chk_shadow, 4, 0)
        
        self.chk_glow = QCheckBox("Свечение")
        self.chk_glow.clicked.connect(self.update_text_properties)
        type_grid.addWidget(self.chk_glow, 4, 1)
        
        self.chk_gradient = QCheckBox("Градиент")
        self.chk_gradient.clicked.connect(self.update_text_properties)
        type_grid.addWidget(self.chk_gradient, 5, 0)
        
        self.chk_vertical = QCheckBox("Вертикально")
        self.chk_vertical.clicked.connect(self.update_text_properties)
        type_grid.addWidget(self.chk_vertical, 5, 1)
        
        self.btn_gradient_color2 = QPushButton("Цвет градиента 2")
        self.btn_gradient_color2.clicked.connect(self.choose_gradient_color2)
        self.btn_gradient_color2.setStyleSheet(f"background-color: {self.gradient_color2.name()};")
        type_grid.addWidget(self.btn_gradient_color2, 6, 0, 1, 2)
        type_ly.addLayout(type_grid)
        
        align_ly = QHBoxLayout()
        self.btn_align_left = QPushButton("Влево")
        self.btn_align_left.clicked.connect(lambda: self.set_text_alignment(Qt.AlignLeft))
        align_ly.addWidget(self.btn_align_left)
        self.btn_align_center = QPushButton("Центр")
        self.btn_align_center.clicked.connect(lambda: self.set_text_alignment(Qt.AlignCenter))
        align_ly.addWidget(self.btn_align_center)
        self.btn_align_right = QPushButton("Вправо")
        self.btn_align_right.clicked.connect(lambda: self.set_text_alignment(Qt.AlignRight))
        align_ly.addWidget(self.btn_align_right)
        type_ly.addLayout(align_ly)
        
        type_ly.addStretch()
        self.tabs.addTab(tab_type, "Тайп")
        
        # Tab 3: AI Translate & Auto-OCR
        tab_ai = QWidget()
        ai_ly = QVBoxLayout(tab_ai)
        
        self.btn_auto_translate_all = QPushButton("Перевести всю главу / страницу (ИИ)")
        self.btn_auto_translate_all.clicked.connect(self.run_ocr_and_translate_all_bubbles)
        ai_ly.addWidget(self.btn_auto_translate_all)
        
        ai_ly.addWidget(QLabel("Оригинальный язык:"))
        self.combo_src = QComboBox()
        self.combo_src.addItems(["Авто", "Японский (ja)", "Корейский (ko)", "Английский (en)"])
        ai_ly.addWidget(self.combo_src)
        
        ai_ly.addWidget(QLabel("Распознанный текст:"))
        self.txt_ocr = QTextEdit()
        self.txt_ocr.setPlaceholderText("Оригинал текста после выделения бабла...")
        ai_ly.addWidget(self.txt_ocr)
        
        ai_ly.addWidget(QLabel("Перевод на русский:"))
        self.txt_trans = QTextEdit()
        self.txt_trans.setPlaceholderText("Отредактируйте перевод...")
        ai_ly.addWidget(self.txt_trans)
        
        self.btn_paste_trans = QPushButton("Вставить в выделенный бабл")
        self.btn_paste_trans.clicked.connect(self.apply_translation_to_bubble)
        ai_ly.addWidget(self.btn_paste_trans)
        ai_ly.addStretch()
        self.tabs.addTab(tab_ai, "ИИ-Перевод")
        
        # Tab 4: Undo/Redo History & General Settings
        tab_settings = QWidget()
        settings_ly = QVBoxLayout(tab_settings)
        
        settings_ly.addWidget(QLabel("История изменений (Ctrl+Z / Ctrl+Y):"))
        self.list_history = QListWidget()
        self.list_history.itemClicked.connect(self.on_history_item_clicked)
        settings_ly.addWidget(self.list_history)
        
        btn_hist_ly = QHBoxLayout()
        self.btn_undo = QPushButton("Отменить")
        self.btn_undo.clicked.connect(self.undo)
        btn_hist_ly.addWidget(self.btn_undo)
        self.btn_redo = QPushButton("Повторить")
        self.btn_redo.clicked.connect(self.redo)
        btn_hist_ly.addWidget(self.btn_redo)
        settings_ly.addLayout(btn_hist_ly)
        
        settings_ly.addWidget(QLabel("Прозрачность клина по умолчанию (%):"))
        self.spin_default_opacity = QSpinBox()
        self.spin_default_opacity.setRange(10, 100)
        self.spin_default_opacity.setValue(100)
        self.spin_default_opacity.valueChanged.connect(self.change_default_clean_opacity)
        settings_ly.addWidget(self.spin_default_opacity)
        
        settings_ly.addStretch()
        self.tabs.addTab(tab_settings, "Настройки")
        
        sidebar_layout.addWidget(self.tabs)
        main_splitter.addWidget(sidebar)
        main_splitter.setSizes([55, 945, 300])

    def setup_canvas_signals(self):
        self.view.drawing_started.connect(self.on_canvas_draw_start)
        self.view.drawing_moved.connect(self.on_canvas_draw_move)
        self.view.drawing_ended.connect(self.on_canvas_draw_end)
        self.view.selection_made.connect(self.on_selection_rectangle_drawn)
        self.scene.selectionChanged.connect(self.on_scene_selection_changed)

    def setup_shortcuts(self):
        self.shortcut_undo = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.shortcut_undo.activated.connect(self.undo)
        
        self.shortcut_redo = QShortcut(QKeySequence("Ctrl+Y"), self)
        self.shortcut_redo.activated.connect(self.redo)
        self.shortcut_redo_alt = QShortcut(QKeySequence("Ctrl+Shift+Z"), self)
        self.shortcut_redo_alt.activated.connect(self.redo)

        # Быстрый выбор инструментов
        self.shortcut_tool_pan = QShortcut(QKeySequence("H"), self)
        self.shortcut_tool_pan.activated.connect(lambda: self.select_tool("pan"))

        self.shortcut_tool_brush = QShortcut(QKeySequence("B"), self)
        self.shortcut_tool_brush.activated.connect(lambda: self.select_tool("brush"))

        self.shortcut_tool_eraser = QShortcut(QKeySequence("E"), self)
        self.shortcut_tool_eraser.activated.connect(lambda: self.select_tool("eraser"))

        self.shortcut_tool_inpaint = QShortcut(QKeySequence("I"), self)
        self.shortcut_tool_inpaint.activated.connect(lambda: self.select_tool("inpaint"))

        self.shortcut_tool_select = QShortcut(QKeySequence("S"), self)
        self.shortcut_tool_select.activated.connect(lambda: self.select_tool("select"))

        self.shortcut_tool_restore = QShortcut(QKeySequence("R"), self)
        self.shortcut_tool_restore.activated.connect(lambda: self.select_tool("restore"))

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls:
            file_path = urls[0].toLocalFile()
            if os.path.isdir(file_path):
                self.load_directory(file_path)
            else:
                self.load_any_file(file_path)
    def select_tool(self, tool):
        self.view.set_tool(tool)
        self.btn_pan.setChecked(tool == "pan")
        self.btn_brush.setChecked(tool == "brush")
        self.btn_eraser.setChecked(tool == "eraser")
        self.btn_inpaint.setChecked(tool == "inpaint")
        self.btn_restore.setChecked(tool == "restore")
        self.btn_select.setChecked(tool == "select")
        self.btn_add_bubble_tool.setChecked(tool == "draw_bubble")
        if tool == "select":
            # Предзагрузка ИИ-детектора при выборе инструмента «Выделение»
            if self.ocr_manager.comic_detector.session is None:
                self.statusBar().showMessage("Загрузка ИИ-модели ComicTextDetector...")
                QApplication.processEvents()
                if self.ocr_manager.comic_detector.load():
                    self.statusBar().showMessage("ИИ-модель ComicTextDetector успешно загружена!", 3000)
                else:
                    # Fallback: загружаем CRAFT EasyOCR
                    if self.ocr_manager.detector_reader is None:
                        self.statusBar().showMessage("Загрузка ИИ-модели CRAFT...")
                        QApplication.processEvents()
                        self.ocr_manager.get_detector_reader()
                        self.statusBar().showMessage("ИИ-модель CRAFT загружена!", 3000)

    # --- UNDO / REDO HISTORY HANDLING ---
    def undo(self):
        if self.undo_stack.canUndo():
            self.undo_stack.undo()
            self.sync_history_list()

    def redo(self):
        if self.undo_stack.canRedo():
            self.undo_stack.redo()
            self.sync_history_list()

    def sync_history_list(self):
        self.list_history.clear()
        count = self.undo_stack.count()
        curr_idx = self.undo_stack.index()
        
        for i in range(count):
            cmd = self.undo_stack.command(i)
            item = QListWidgetItem(cmd.text())
            if i >= curr_idx:
                item.setForeground(QColor(113, 113, 122))
            else:
                item.setForeground(QColor(244, 244, 245))
            self.list_history.addItem(item)
            
        self.list_history.setCurrentRow(curr_idx - 1)

    def on_history_item_clicked(self, item):
        row = self.list_history.row(item)
        self.undo_stack.setIndex(row + 1)
        self.sync_history_list()

    # --- LAYERS MANAGEMENT ---
    def sync_layers_list(self):
        self.list_layers.clear()
        for idx in reversed(range(len(self.layers))):
            layer = self.layers[idx]
            vis_str = "Показ" if layer.is_visible else "Скрыт"
            item = QListWidgetItem(f"{layer.layer_name} ({vis_str}, Opacity: {int(layer.opacity() * 100)}%)")
            self.list_layers.addItem(item)
            
    def add_blank_layer(self, name="Новый слой"):
        if not self.layers:
            return
        w, h = self.layers[0].width, self.layers[0].height
        layer = LayerItem(w, h, name)
        
        if "клин" in name.lower() or "слой" in name.lower():
            layer.setOpacity(self.default_clean_opacity / 100.0)
            
        self.scene.addItem(layer)
        self.layers.append(layer)
        self.sync_layers_list()
        layer.setZValue(len(self.layers))
        return layer

    def add_image_layer(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Импорт слоя", "", "Изображения (*.png *.jpg *.jpeg)")
        if file_path and self.layers:
            pix = QPixmap(file_path)
            w, h = self.layers[0].width, self.layers[0].height
            layer = LayerItem(w, h, "Импортированное изображение")
            
            painter = QPainter(layer.image)
            painter.drawPixmap(0, 0, pix.scaled(w, h, Qt.KeepAspectRatio))
            painter.end()
            layer.setPixmap(QPixmap.fromImage(layer.image))
            
            self.scene.addItem(layer)
            self.layers.append(layer)
            self.sync_layers_list()
            layer.setZValue(len(self.layers))

    def on_layer_selected(self, item):
        row = self.list_layers.row(item)
        target_idx = len(self.layers) - 1 - row
        if 0 <= target_idx < len(self.layers):
            layer = self.layers[target_idx]
            self.chk_layer_visible.setChecked(layer.is_visible)
            self.slider_layer_opacity.setValue(int(layer.opacity() * 100))

    def get_active_layer(self):
        row = self.list_layers.currentRow()
        if row == -1:
            return self.layers[-1] if self.layers else None
        target_idx = len(self.layers) - 1 - row
        if 0 <= target_idx < len(self.layers):
            return self.layers[target_idx]
        return self.layers[-1] if self.layers else None

    def toggle_active_layer_visibility(self):
        layer = self.get_active_layer()
        if layer:
            layer.is_visible = self.chk_layer_visible.isChecked()
            layer.setVisible(layer.is_visible)
            self.sync_layers_list()

    def set_active_layer_opacity(self, val):
        layer = self.get_active_layer()
        if layer:
            layer.setOpacity(val / 100.0)
            self.sync_layers_list()

    def change_default_clean_opacity(self, val):
        self.default_clean_opacity = val

    # --- IMAGE LOADING ---
    def load_image_or_project(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Открыть файл", "", "Все форматы (*.png *.jpg *.jpeg *.mproj *.psd)"
        )
        if file_path:
            self.load_any_file(file_path)

    def load_any_file(self, file_path):
        if file_path.endswith(".psd"):
            self.import_psd_file(file_path)
        elif file_path.endswith(".mproj"):
            self.load_mproj_file(file_path)
        else:
            self.load_standard_image(file_path)
    def on_original_image_loaded(self):
        if self.original_cv_image is not None:
            self.pristine_cv_image = self.original_cv_image.copy()
            h, w = self.pristine_cv_image.shape[:2]
            self.pristine_qimage = QImage(self.pristine_cv_image.data, w, h, 3*w, QImage.Format_BGR888).copy()
        else:
            self.pristine_cv_image = None
            self.pristine_qimage = None
            
    def check_and_load_lama(self, prompt=True):
        """Проверяет наличие ИИ-модели LaMa MPE, скачивает её при необходимости и загружает сессию."""
        model_dir = "models"
        model_path = os.path.join(model_dir, "inpainting_lama_mpe.ckpt")
        
        if not os.path.exists(model_path):
            if prompt:
                reply = QMessageBox.question(
                    self, 
                    "Загрузка ИИ-модели", 
                    "Для качественной очистки необходимо скачать ИИ-модель LaMa MPE (около 100 МБ), дообученную на манге.\nСкачать её сейчас?",
                    QMessageBox.Yes | QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return False
            
            # Скачиваем с GitHub Releases
            ok = self.download_model_gui("https://github.com/zyddnys/manga-image-translator/releases/download/beta-0.3/inpainting_lama_mpe.ckpt", model_path)
            if not ok:
                QMessageBox.warning(self, "Загрузка отменена", "Будет использован стандартный алгоритм (Navier-Stokes).")
                return False
        if self.lama_inpainter is None:
            try:
                self.statusBar().showMessage("Загрузка ИИ-модели LaМа MPE в память (PyTorch)...")
                QApplication.processEvents()
                self.lama_inpainter = LaMaInpainter(model_path)
                self.statusBar().showMessage("ИИ-модель LaМа MPE готова к работе!", 3000)
            except Exception as e:
                QMessageBox.warning(self, "Ошибка инициализации", f"Не удалось запустить LaMa: {e}\nБудет использован стандартный алгоритм.")
                self.lama_inpainter = None
                return False
        return True
    def load_text_segmenter(self):
        segmenter_path = os.path.join("models", "segmenter.onnx")
        if os.path.exists(segmenter_path):
            try:
                self.text_segmenter = ort.InferenceSession(segmenter_path, providers=["CPUExecutionProvider"])
                print("ИИ-сегментатор текста U-Net успешно загружен!")
            except Exception as e:
                print(f"Ошибка загрузки U-Net сегментатора: {e}")

    def download_model_gui(self, url, dest_path):
        import urllib.request
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        
        dialog = QProgressDialog("Скачивание ИИ-модели очистки (LaMa)...", "Отмена", 0, 100, self)
        dialog.setWindowTitle("Загрузка модели")
        dialog.setWindowModality(Qt.WindowModal)
        dialog.setAutoClose(True)
        dialog.show()
        
        canceled = False
        def progress_callback(block_num, block_size, total_size):
            nonlocal canceled
            if dialog.wasCanceled():
                canceled = True
                return
            if total_size > 0:
                percent = int(block_num * block_size * 100 / total_size)
                dialog.setValue(min(percent, 100))
                QApplication.processEvents()
                
        try:
            urllib.request.urlretrieve(url, dest_path, reporthook=progress_callback)
            if canceled:
                if os.path.exists(dest_path):
                    os.remove(dest_path)
                return False
            return True
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки", f"Не удалось скачать модель: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            return False

    def load_standard_image(self, file_path):
        self.original_cv_image = cv_imread_unicode(file_path)
        if self.original_cv_image is None:
            QMessageBox.critical(self, "Ошибка", "Не удалось прочитать файл изображения.")
            return

        self.on_original_image_loaded()
        self.scene.clear()
        self.layers.clear()
        self.undo_stack.clear()
        self.sync_history_list()
        
        pix = QPixmap(file_path)
        if pix.isNull():
            h, w = self.original_cv_image.shape[:2]
            q_img = QImage(self.original_cv_image.data, w, h, 3*w, QImage.Format_BGR888).copy()
            pix = QPixmap.fromImage(q_img)
            
        bg_layer = LayerItem(pix.width(), pix.height(), "Оригинал")
        painter = QPainter(bg_layer.image)
        painter.drawPixmap(0, 0, pix)
        painter.end()
        bg_layer.setPixmap(QPixmap.fromImage(bg_layer.image))
        self.scene.addItem(bg_layer)
        self.layers.append(bg_layer)
        
        self.add_blank_layer("Слой клина")
        
        self.scene.setSceneRect(0, 0, pix.width(), pix.height())
        self.view.fitInView(bg_layer, Qt.KeepAspectRatio)
        self.sync_layers_list()
        
        self.list_layers.setCurrentRow(0)
        self.select_tool("brush")
    # --- DIRECTORY NAVIGATION ---
    def open_folder_dialog(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Открыть папку главы")
        if dir_path:
            self.load_directory(dir_path)

    def load_directory(self, dir_path):
        self.folder_images = []
        for file in sorted(os.listdir(dir_path)):
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                self.folder_images.append(os.path.join(dir_path, file))
                
        if self.folder_images:
            self.current_folder_index = 0
            self.btn_prev_page.setEnabled(True)
            self.btn_next_page.setEnabled(True)
            self.update_folder_page()
        else:
            QMessageBox.information(self, "Пусто", "В этой папке нет картинок.")

    def update_folder_page(self):
        if 0 <= self.current_folder_index < len(self.folder_images):
            file_path = self.folder_images[self.current_folder_index]
            self.load_any_file(file_path)
            self.lbl_page_info.setText(f"Стр {self.current_folder_index+1} / {len(self.folder_images)}")

    def prev_page(self):
        if self.current_folder_index > 0:
            self.current_folder_index -= 1
            self.update_folder_page()

    def next_page(self):
        if self.current_folder_index < len(self.folder_images) - 1:
            self.current_folder_index += 1
            self.update_folder_page()

    # --- DRAWING EVENTS ---
    def on_canvas_draw_start(self, pos):
        active_layer = self.get_active_layer()
        if not active_layer:
            return
            
        if self.layers and active_layer == self.layers[0] and self.view.tool not in ["restore", "inpaint"]:
            self.statusBar().showMessage("Предупреждение: Рисование на оригинальном слое заблокировано! Выберите другой слой.", 3000)
            return
            
        if not active_layer.is_visible:
            active_layer.is_visible = True
            active_layer.setVisible(True)
            self.sync_layers_list()
            self.statusBar().showMessage("Активный слой автоматически сделан видимым.", 2000)
            
        if self.view.tool in ["brush", "eraser", "inpaint", "restore"]:
            self.draw_start_image = active_layer.image.copy()
            active_layer.draw_line(pos.toPoint(), pos.toPoint(), self.view.tool, self.brush_color, self.spin_brush_size.value(), self.pristine_qimage)

    def on_canvas_draw_move(self, start, end):
        active_layer = self.get_active_layer()
        if active_layer and self.view.tool in ["brush", "eraser", "inpaint", "restore"]:
            active_layer.draw_line(start.toPoint(), end.toPoint(), self.view.tool, self.brush_color, self.spin_brush_size.value(), self.pristine_qimage)

    def on_canvas_draw_end(self):
        active_layer = self.get_active_layer()
        if active_layer and self.draw_start_image is not None and self.view.tool in ["brush", "eraser", "inpaint", "restore"]:
            if self.view.tool == "inpaint":
                # Воспроизводим поведение Spot Healing Brush (Кисть удаления)
                h, w = active_layer.image.height(), active_layer.image.width()
                
                curr_qimg = active_layer.image.convertToFormat(QImage.Format_ARGB32)
                start_qimg = self.draw_start_image.convertToFormat(QImage.Format_ARGB32)
                
                curr_ptr = curr_qimg.bits()
                start_ptr = start_qimg.bits()
                
                curr_arr = np.array(curr_ptr).reshape(h, w, 4)
                start_arr = np.array(start_ptr).reshape(h, w, 4)
                
                # Ищем где изменился альфа-канал или цвета (красный штрих)
                diff = np.any(curr_arr != start_arr, axis=-1)
                
                if np.any(diff):
                    mask = (diff.astype(np.uint8) * 255)
                    
                    if self.original_cv_image is not None:
                        # Подгружаем LaMa при необходимости
                        self.check_and_load_lama(prompt=False)
                        
                        try:
                            self.statusBar().showMessage("ИИ стирает и дорисовывает выделенную кистью область...")
                            QApplication.processEvents()
                            
                            h_bg, w_bg = self.layers[0].image.height(), self.layers[0].image.width()
                            
                            # Берём чистый BGR до мазка
                            if active_layer == self.layers[0]:
                                start_bgr_qimg = self.draw_start_image.convertToFormat(QImage.Format_BGR888)
                                start_bgr_ptr = start_bgr_qimg.bits()
                                bg_cv = np.array(start_bgr_ptr).reshape(h_bg, w_bg, 3).copy()
                            else:
                                bg_qimg = self.layers[0].image.convertToFormat(QImage.Format_BGR888)
                                bg_ptr = bg_qimg.bits()
                                bg_cv = np.array(bg_ptr).reshape(h_bg, w_bg, 3).copy()
                            
                            # Находим Bounding Box мазка (где нарисовал пользователь)
                            y_indices, x_indices = np.where(mask > 127)
                            y0, y1 = y_indices.min(), y_indices.max()
                            x0, x1 = x_indices.min(), x_indices.max()
                            
                            # Добавляем падинг 48 пикселей во все стороны для контекста
                            pad = 48
                            crop_y0 = max(0, y0 - pad)
                            crop_y1 = min(h_bg, y1 + pad)
                            crop_x0 = max(0, x0 - pad)
                            crop_x1 = min(w_bg, x1 + pad)
                            
                            crop_img = bg_cv[crop_y0:crop_y1, crop_x0:crop_x1].copy()
                            crop_mask = mask[crop_y0:crop_y1, crop_x0:crop_x1].copy()
                            
                            # Запуск Inpainting на КРОПЕ
                            if self.lama_inpainter is not None:
                                crop_inpainted = self.lama_inpainter.inpaint(crop_img, crop_mask)
                            else:
                                crop_inpainted = cv2.inpaint(crop_img, crop_mask, 3, cv2.INPAINT_TELEA)
                                
                            # Вставляем кроп обратно в исходное изображение
                            inpainted = bg_cv.copy()
                            inpainted[crop_y0:crop_y1, crop_x0:crop_x1] = crop_inpainted
                            
                            bg_before_cv = self.original_cv_image.copy()
                            self.original_cv_image = inpainted.copy()
                            
                            # Конвертируем обратно в QImage
                            inpainted_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
                            q_img = QImage(inpainted_rgb.data, w_bg, h_bg, w_bg * 3, QImage.Format_RGB888).copy()
                            
                            bg_before = self.layers[0].image.copy()
                            self.layers[0].image = q_img
                            self.layers[0].setPixmap(QPixmap.fromImage(q_img))
                            
                            # Фиксируем в Undo
                            desc = "ИИ Кисть удаления"
                            cmd = UndoPaintCommand(self.layers[0], bg_before, self.layers[0].image, desc, self)
                            cmd.set_cv_images(bg_before_cv, self.original_cv_image)
                            self.undo_stack.push(cmd)
                            
                        except Exception as e:
                            print(f"Inpaint brush error: {e}")
                            self.statusBar().showMessage(f"Ошибка ИИ: {e}", 3000)
                
                # Если рисовали на прозрачном слое — очищаем его от красного штриха
                if active_layer != self.layers[0]:
                    active_layer.image = self.draw_start_image.copy()
                    active_layer.setPixmap(QPixmap.fromImage(active_layer.image))
                else:
                    # Если рисовали на оригинальном слое, то красная линия заменяется на результат inpaint
                    pass
                    
                self.draw_start_image = None
                self.sync_history_list()
                self.statusBar().showMessage("Область успешно очищена ИИ!", 3000)
                return
            
            # Стандартная обработка для остальных инструментов (кисть, ластик, восстановление)
            if active_layer == self.layers[0]:
                h, w = active_layer.image.height(), active_layer.image.width()
                temp_qimg = active_layer.image.convertToFormat(QImage.Format_BGR888)
                ptr = temp_qimg.bits()
                arr = np.array(ptr).reshape(h, w, 3)
                self.original_cv_image = arr.copy()
                
            desc = "Восстановление" if self.view.tool == "restore" else ("Кисть клина" if self.view.tool == "brush" else ("Ластик" if self.view.tool == "eraser" else "ИИ-Маска"))
            cmd = UndoPaintCommand(active_layer, self.draw_start_image, active_layer.image, desc, self)
            self.undo_stack.push(cmd)
            self.draw_start_image = None
            self.sync_history_list()

    def choose_brush_color(self):
        col = QColorDialog.getColor(self.brush_color, self, "Выбрать цвет")
        if col.isValid():
            self.brush_color = col
            self.btn_brush_color.setStyleSheet(f"background-color: {col.name()};")

    # --- STANDARD SELECTION TOOL OPERATIONS ---
    def on_selection_rectangle_drawn(self, rect, is_shift_pressed=False):
        if self.view.tool == "select":
            # Если не нажат Shift, удаляем все предыдущие нарисованные баблы
            if not is_shift_pressed:
                for item in self.scene.items():
                    if isinstance(item, InteractiveBubbleItem) and getattr(item, "is_manual", False):
                        self.scene.removeItem(item)
            
            bubble = InteractiveBubbleItem(rect)
            bubble.is_manual = True
            # Снимаем выделение со всех остальных и выделяем только новый
            self.scene.clearSelection()
            bubble.setSelected(True)
            self.scene.addItem(bubble)
            
        elif self.view.tool == "draw_bubble":
            bubble = InteractiveBubbleItem(rect)
            bubble.is_manual = True
            self.scene.clearSelection()
            bubble.setSelected(True)
            self.scene.addItem(bubble)
            self.select_tool("pan")

    def clear_selection_area(self):
        active_layer = self.get_active_layer()
        if not active_layer:
            return
            
        bubbles_to_clear = [item for item in self.scene.selectedItems() if isinstance(item, InteractiveBubbleItem)]
        if not bubbles_to_clear:
            QMessageBox.information(self, "Инфо", "Сначала выделите бабл(ы) для очистки.")
            return
            
        self.draw_start_image = active_layer.image.copy()
        painter = QPainter(active_layer.image)
        painter.setCompositionMode(QPainter.CompositionMode_Clear)
        
        for bubble in bubbles_to_clear:
            painter.fillRect(bubble.rect(), Qt.transparent)
            
        painter.end()
        active_layer.setPixmap(QPixmap.fromImage(active_layer.image))
        
        cmd = UndoPaintCommand(active_layer, self.draw_start_image, active_layer.image, "Очистка выделения", self)
        self.undo_stack.push(cmd)
        self.sync_history_list()

    def inpaint_selection_area(self):
        if self.original_cv_image is None:
            return
            
        bubbles_to_inpaint = [item for item in self.scene.selectedItems() if isinstance(item, InteractiveBubbleItem)]
        if not bubbles_to_inpaint:
            QMessageBox.information(self, "Инфо", "Выберите бабл(ы) для ИИ очистки (Ластик).")
            return
            
        bg_before = self.layers[0].image.copy()
        cv_before = self.original_cv_image.copy()
        self.check_and_load_lama()
        self.statusBar().showMessage("Применяется ИИ очистка LaMa...")
        QApplication.processEvents()
        
        for bubble in bubbles_to_inpaint:
            scene_rect = bubble.mapToScene(bubble.rect()).boundingRect()
            self.original_cv_image = smart_inpaint_rect(
                self.original_cv_image, 
                scene_rect, 
                dilation_pixels=6,
                lama_inpainter=self.lama_inpainter,
                text_segmenter=self.text_segmenter
            )
        self.statusBar().showMessage("Готово!", 3000)
        
        full_h, full_w = self.original_cv_image.shape[:2]
        q_img = QImage(self.original_cv_image.data, full_w, full_h, 3*full_w, QImage.Format_BGR888).copy()
        self.layers[0].image = q_img
        self.layers[0].setPixmap(QPixmap.fromImage(q_img))
        
        cmd = UndoPaintCommand(self.layers[0], bg_before, self.layers[0].image, "ИИ-очистка баблов", self)
        cmd.set_cv_images(cv_before, self.original_cv_image)
        self.undo_stack.push(cmd)
        
        for bubble in bubbles_to_inpaint:
            if getattr(bubble, "is_manual", False):
                self.scene.removeItem(bubble)
                
        self.sync_history_list()
        QMessageBox.information(self, "Очистка готова", "Выделенные области успешно очищены!")
    # --- BUBBLE DETECTOR & ADVANCED OBLITERATION ---
    def detect_manga_bubbles(self):
        """Uses EasyOCR CRAFT text detector to locate boxes, falling back to OpenCV contours."""
        if self.original_cv_image is None:
            QMessageBox.warning(self, "Ошибка", "Сначала откройте изображение.")
            return
            
        self.statusBar().showMessage("Поиск текстовых баблов (ИИ ComicTextDetector)...")
        QApplication.processEvents()
        
        for item in list(self.scene.items()):
            if isinstance(item, InteractiveBubbleItem):
                self.scene.removeItem(item)
                
        qrects = []
        try:
            src_langs = {"Авто": "auto", "Японский (ja)": "ja", "Корейский (ko)": "ko", "Английский (en)": "en"}
            src = src_langs.get(self.combo_src.currentText(), "auto")
            lang_code = src if src != 'auto' else 'ja'
            
            # ComicTextDetector (ONNX) is primary, EasyOCR CRAFT is fallback
            qrects = self.ocr_manager.detect_bubbles_ai(self.original_cv_image, lang_code)
        except Exception as e:
            print(f"Detection error: {e}")
                
        if not qrects:
            # Fallback contour detector if AI not available or found nothing
            gray = cv2.cvtColor(self.original_cv_image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = w * h
                if 600 < area < (self.original_cv_image.shape[0] * self.original_cv_image.shape[1] * 0.4):
                    aspect_ratio = float(w) / h
                    if 0.3 < aspect_ratio < 3.0:
                        qrects.append(QRectF(x, y, w, h))
                        
        for r in qrects:
            bubble = InteractiveBubbleItem(r)
            self.scene.addItem(bubble)
            
        self.statusBar().showMessage(f"Найдено баблов: {len(qrects)}", 4000)

    def delete_selected_bubble(self):
        sel = self.scene.selectedItems()
        for item in sel:
            if isinstance(item, InteractiveBubbleItem):
                self.scene.removeItem(item)

    def clear_all_detected_bubbles(self):
        """Cleans text inside all bubbles using high-precision CC segmentation and dilation."""
        if self.original_cv_image is None:
            return
            
        bubbles = [item for item in self.scene.items() if isinstance(item, InteractiveBubbleItem)]
        if not bubbles:
            QMessageBox.information(self, "Инфо", "На холсте нет баблов.")
            return
        self.check_and_load_lama()
        self.statusBar().showMessage("Высокоточная очистка текста внутри баблов...")
        QApplication.processEvents()
        
        bg_before = self.layers[0].image.copy()
        cv_before = self.original_cv_image.copy()
        self.original_cv_image, cleaned_count = smart_clean_bubbles(
            self.original_cv_image, 
            bubbles, 
            dilation_pixels=5, 
            lama_inpainter=self.lama_inpainter,
            text_segmenter=self.text_segmenter
        )
        full_h, full_w = self.original_cv_image.shape[:2]
        q_img = QImage(self.original_cv_image.data, full_w, full_h, 3*full_w, QImage.Format_BGR888).copy()
        self.layers[0].image = q_img
        self.layers[0].setPixmap(QPixmap.fromImage(q_img))
        
        cmd = UndoPaintCommand(self.layers[0], bg_before, self.layers[0].image, f"Очищено баблов: {cleaned_count}", self)
        cmd.set_cv_images(cv_before, self.original_cv_image)
        self.undo_stack.push(cmd)
        
        self.statusBar().showMessage(f"Успешно очищено баблов: {cleaned_count}!", 4000)
        self.sync_history_list()

    def run_ocr_and_translate_all_bubbles(self):
        bubbles = [item for item in self.scene.items() if isinstance(item, InteractiveBubbleItem)]
        if not bubbles:
            QMessageBox.information(self, "Инфо", "Сначала выделите баблы.")
            return
        src_langs = {"Авто": "auto", "Японский (ja)": "ja", "Корейский (ko)": "ko", "Английский (en)": "en"}
        src = src_langs.get(self.combo_src.currentText(), "auto")
        
        lang_code = src if src != 'auto' else 'ja'
        reader = self.ocr_manager.get_transcribe_reader(lang_code)
        if not EASYOCR_AVAILABLE or reader is None:
            QMessageBox.warning(self, "Ошибка", "ИИ-распознаватель EasyOCR недоступен.")
            return
            
        self.statusBar().showMessage("Автоперевод всех баблов...")
        QApplication.processEvents()
        
        translated_count = 0
        for bubble in bubbles:
            rect = bubble.rect()
            pos = bubble.scenePos()
            x = int(pos.x() + rect.x())
            y = int(pos.y() + rect.y())
            w = int(rect.width())
            h = int(rect.height())
            
            full_h, full_w = self.original_cv_image.shape[:2]
            x = max(0, min(x, full_w - 1))
            y = max(0, min(y, full_h - 1))
            w = min(w, full_w - x)
            h = min(h, full_h - y)
            
            crop = self.original_cv_image[y:y+h, x:x+w]
            if crop.size == 0:
                continue
                
            text_detected = ""
            if lang_code == 'ja':
                mocr = self.ocr_manager.get_manga_ocr()
                if mocr:
                    try:
                        from PIL import Image
                        pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        text_detected = mocr(pil_img)
                    except Exception as e:
                        print(f"Manga-OCR error during translation: {e}")
                        
            if not text_detected.strip():
                rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                res = reader.readtext(rgb)
                text_detected = " ".join([r[1] for r in res])
                
            if text_detected.strip():
                translated = translate_google(text_detected, src, "ru")
                
                item = TypesetTextItem(translated)
                font = QFont(self.combo_font.currentText())
                font.setPointSize(self.spin_font_size.value())
                item.setFont(font)
                
                item.font_color = self.text_color
                item.outline_color = self.outline_color
                item.outline_width = self.spin_outline.value()
                
                self.scene.addItem(item)
                item.setPos(x + 5, y + 5)
                item.setTextWidth(w - 10)
                
                doc = item.document()
                opt = doc.defaultTextOption()
                opt.setAlignment(Qt.AlignCenter)
                doc.setDefaultTextOption(opt)
                
                metrics = QFontMetrics(item.font())
                lines_est = math.ceil(metrics.horizontalAdvance(translated) / (w - 10))
                if lines_est > 0:
                    h_est = lines_est * metrics.lineSpacing()
                    if h_est > (h - 10):
                        curr_size = item.font().pointSize()
                        new_size = max(6, int(curr_size * (h - 10) / h_est))
                        font.setPointSize(new_size)
                        item.setFont(font)
                        
                cmd = UndoAddTextCommand(self.scene, item, self)
                self.undo_stack.push(cmd)
                translated_count += 1
                
        self.statusBar().showMessage(f"Успешно переведено баблов: {translated_count}", 4000)
        self.sync_history_list()

    def apply_translation_to_bubble(self):
        text = self.txt_trans.toPlainText()
        if not text:
            return
            
        self.add_text_item(text)
        if hasattr(self, 'last_bubble_rect') and self.current_text_item:
            x, y, w, h = self.last_bubble_rect
            self.current_text_item.setPos(x + 5, y + 5)
            self.current_text_item.setTextWidth(w - 10)
            self.set_text_alignment(Qt.AlignCenter)
            
            metrics = QFontMetrics(self.current_text_item.font())
            lines_est = math.ceil(metrics.horizontalAdvance(text) / (w - 10))
            if lines_est > 0:
                h_est = lines_est * metrics.lineSpacing()
                if h_est > (h - 10):
                    curr_size = self.current_text_item.font().pointSize()
                    new_size = max(6, int(curr_size * (h - 10) / h_est))
                    self.spin_font_size.setValue(new_size)
                    self.update_text_properties()

    def add_text_item(self, text="Текст"):
        if isinstance(text, bool):
            text = "Текст"
        if not self.layers:
            return
        item = TypesetTextItem(text)
        font = QFont(self.combo_font.currentText())
        font.setPointSize(self.spin_font_size.value())
        item.setFont(font)
        item.font_color = self.text_color
        item.outline_color = self.outline_color
        item.outline_width = self.spin_outline.value()
        item.set_vertical(self.chk_vertical.isChecked())
        item.has_shadow = self.chk_shadow.isChecked()
        item.has_glow = self.chk_glow.isChecked()
        item.use_gradient = self.chk_gradient.isChecked()
        item.gradient_color2 = self.gradient_color2
        
        self.scene.addItem(item)
        center = self.view.mapToScene(self.view.viewport().rect().center())
        item.setPos(center)
        item.setSelected(True)
        self.current_text_item = item
        
        cmd = UndoAddTextCommand(self.scene, item, self)
        self.undo_stack.push(cmd)
        self.sync_history_list()

    def on_scene_selection_changed(self):
        sel = self.scene.selectedItems()
        if sel and isinstance(sel[0], TypesetTextItem):
            self.current_text_item = sel[0]
            self.spin_font_size.setValue(self.current_text_item.font().pointSize())
            self.spin_outline.setValue(self.current_text_item.outline_width)
            self.chk_vertical.setChecked(self.current_text_item.is_vertical)
            self.chk_shadow.setChecked(self.current_text_item.has_shadow)
            self.chk_glow.setChecked(self.current_text_item.has_glow)
            self.chk_gradient.setChecked(self.current_text_item.use_gradient)
            
            self.text_color = self.current_text_item.font_color
            self.outline_color = self.current_text_item.outline_color
            self.gradient_color2 = self.current_text_item.gradient_color2
            
            self.btn_txt_color.setStyleSheet(f"background-color: {self.text_color.name()};")
            self.btn_outline_color.setStyleSheet(f"background-color: {self.outline_color.name()};")
            self.btn_gradient_color2.setStyleSheet(f"background-color: {self.gradient_color2.name()};")
            
            idx = self.combo_font.findText(self.current_text_item.font().family())
            if idx != -1:
                self.combo_font.setCurrentIndex(idx)

    def update_text_properties(self):
        if self.current_text_item:
            font = self.current_text_item.font()
            font.setFamily(self.combo_font.currentText())
            font.setPointSize(self.spin_font_size.value())
            self.current_text_item.setFont(font)
            
            self.current_text_item.outline_width = self.spin_outline.value()
            self.current_text_item.is_vertical = self.chk_vertical.isChecked()
            self.current_text_item.has_shadow = self.chk_shadow.isChecked()
            self.current_text_item.has_glow = self.chk_glow.isChecked()
            self.current_text_item.use_gradient = self.chk_gradient.isChecked()
            
            self.current_text_item.font_color = self.text_color
            self.current_text_item.outline_color = self.outline_color
            self.current_text_item.gradient_color2 = self.gradient_color2
            
            self.current_text_item.update()

    def set_text_alignment(self, align):
        if self.current_text_item:
            doc = self.current_text_item.document()
            option = doc.defaultTextOption()
            option.setAlignment(align)
            doc.setDefaultTextOption(option)
            self.current_text_item.update()

    def choose_text_color(self):
        col = QColorDialog.getColor(self.text_color, self, "Цвет шрифта")
        if col.isValid():
            self.text_color = col
            self.btn_txt_color.setStyleSheet(f"background-color: {col.name()};")
            if self.current_text_item:
                self.current_text_item.font_color = col
                self.current_text_item.update()

    def choose_outline_color(self):
        col = QColorDialog.getColor(self.outline_color, self, "Цвет контура")
        if col.isValid():
            self.outline_color = col
            self.btn_outline_color.setStyleSheet(f"background-color: {col.name()};")
            if self.current_text_item:
                self.current_text_item.outline_color = col
                self.current_text_item.update()

    def choose_gradient_color2(self):
        col = QColorDialog.getColor(self.gradient_color2, self, "Цвет градиента 2")
        if col.isValid():
            self.gradient_color2 = col
            self.btn_gradient_color2.setStyleSheet(f"background-color: {col.name()};")
            if self.current_text_item:
                self.current_text_item.gradient_color2 = col
                self.current_text_item.update()

    def open_font_manager(self):
        dlg = FontManagerDialog(self.project_fonts, self)
        if dlg.exec() == QDialog.Accepted:
            self.combo_font.clear()
            self.combo_font.addItems(self.project_fonts)

    def open_waifu_dialog(self):
        if self.original_cv_image is None:
            QMessageBox.warning(self, "Ошибка", "Сначала откройте изображение.")
            return
        dlg = WaifuDialog(self.original_cv_image, self)
        if dlg.exec() == QDialog.Accepted and dlg.upscaled_image is not None:
            bg_before = self.layers[0].image.copy()
            cv_before = self.original_cv_image.copy() if self.original_cv_image is not None else None
            self.original_cv_image = dlg.upscaled_image
            self.on_original_image_loaded()
            
            h, w = self.original_cv_image.shape[:2]
            q_img = QImage(self.original_cv_image.data, w, h, 3*w, QImage.Format_BGR888).copy()
            
            self.layers[0].width = w
            self.layers[0].height = h
            self.layers[0].image = q_img
            self.layers[0].setPixmap(QPixmap.fromImage(q_img))
            
            if len(self.layers) > 1:
                clean_layer = self.layers[1]
                clean_layer.width = w
                clean_layer.height = h
                clean_layer.image = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
                clean_layer.image.fill(Qt.transparent)
                clean_layer.setPixmap(QPixmap.fromImage(clean_layer.image))
                
            self.scene.setSceneRect(0, 0, w, h)
            self.view.fitInView(self.layers[0], Qt.KeepAspectRatio)
            
            cmd = UndoPaintCommand(self.layers[0], bg_before, self.layers[0].image, "Вайфу Апскейл", self)
            cmd.set_cv_images(cv_before, self.original_cv_image)
            self.undo_stack.push(cmd)
            self.sync_history_list()

    # --- MANGA / MANHWA FUNCTIONS ---
    def toggle_mode(self, index):
        is_manhwa = (index == 1)
        self.manhwa_bar_widget.setVisible(is_manhwa)
        if is_manhwa:
            self.chk_grayscale.setChecked(False)
            self.toggle_grayscale()
            QMessageBox.information(self, "Режим Манхва", "Включен режим склейки и нарезки длинной ленты манхвы.")
        else:
            self.view.cutting_lines.clear()
            self.view.draw_cutting_lines()
            QMessageBox.information(self, "Режим Манга", "Режим работы с отдельными страницами манги.")

    def toggle_grayscale(self):
        if self.original_cv_image is None or not self.layers:
            return
            
        if self.chk_grayscale.isChecked():
            gray = cv2.cvtColor(self.original_cv_image, cv2.COLOR_BGR2GRAY)
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            h, w = gray_bgr.shape[:2]
            q_img = QImage(gray_bgr.data, w, h, 3*w, QImage.Format_BGR888).copy()
            self.layers[0].setPixmap(QPixmap.fromImage(q_img))
        else:
            h, w = self.original_cv_image.shape[:2]
            q_img = QImage(self.original_cv_image.data, w, h, 3*w, QImage.Format_BGR888).copy()
            self.layers[0].setPixmap(QPixmap.fromImage(q_img))

    def stitch_manhwa_pages(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Выберите страницы манхвы для склейки", "", "Изображения (*.png *.jpg *.jpeg)"
        )
        if not files:
            return
            
        images = []
        for f in files:
            img = cv_imread_unicode(f)
            if img is not None:
                images.append(img)
                
        if not images:
            return
            
        target_w = images[0].shape[1]
        resized_imgs = []
        for img in images:
            h, w = img.shape[:2]
            if w != target_w:
                new_h = int(h * target_w / w)
                resized = cv2.resize(img, (target_w, new_h), interpolation=cv2.INTER_CUBIC)
                resized_imgs.append(resized)
            else:
                resized_imgs.append(img)
                
        stitched = np.vstack(resized_imgs)
        self.original_cv_image = stitched
        self.on_original_image_loaded()
        
        h, w = stitched.shape[:2]
        q_img = QImage(stitched.data, w, h, 3*w, QImage.Format_BGR888).copy()
        
        self.scene.clear()
        self.layers.clear()
        self.undo_stack.clear()
        self.sync_history_list()
        
        bg = LayerItem(w, h, "Склейка ");
        bg.image = q_img
        bg.setPixmap(QPixmap.fromImage(q_img))
        self.scene.addItem(bg)
        self.layers.append(bg)
        
        self.add_blank_layer("Слой клина")
        self.scene.setSceneRect(0, 0, w, h)
        self.view.fitInView(bg, Qt.KeepAspectRatio)
        self.sync_layers_list()
        QMessageBox.information(self, "Готово", f"Склеено страниц: {len(images)}. Высота полосы: {h}px.")

    def toggle_cut_mode(self):
        if self.btn_cut_mode.isChecked():
            self.select_tool("cut")
            QMessageBox.information(self, "Нарезка", "Кликайте на холст, чтобы установить горизонтальные линии разреза.")
        else:
            self.select_tool("pan")

    def slice_manhwa_pages(self):
        if self.original_cv_image is None:
            return
            
        out_dir = QFileDialog.getExistingDirectory(self, "Выбрать папку для сохранения страниц")
        if not out_dir:
            return
            
        h, w = self.original_cv_image.shape[:2]
        full_scene_img = QImage(self.scene.sceneRect().size().toSize(), QImage.Format_ARGB32)
        full_scene_img.fill(Qt.transparent)
        
        painter = QPainter(full_scene_img)
        self.scene.render(painter)
        painter.end()
        
        rgba_img = full_scene_img.convertToFormat(QImage.Format_RGBA8888)
        width = rgba_img.width()
        height = rgba_img.height()
        bytes_per_line = rgba_img.bytesPerLine()
        
        ptr = rgba_img.constBits()
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, bytes_per_line // 4, 4))
        if bytes_per_line // 4 != width:
            arr = arr[:, :width, :]
        arr = arr.copy()
        
        full_cv_img = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        
        y_cuts = sorted(list(set([0] + [int(y) for y in self.view.cutting_lines] + [h])))
        if len(y_cuts) <= 2:
            num_pages = 10
            page_h = h // num_pages
            y_cuts = [i * page_h for i in range(num_pages)] + [h]
            
        for i in range(len(y_cuts) - 1):
            y_start = y_cuts[i]
            y_end = y_cuts[i+1]
            if y_end - y_start < 20:
                continue
            slice_img = full_cv_img[y_start:y_end, 0:w]
            cv_imwrite_unicode(os.path.join(out_dir, f"page_{i+1:03d}.png"), slice_img)
            
        QMessageBox.information(self, "Экспорт завершен", f"Манхва успешно нарезана на {len(y_cuts)-1} страниц.")
        self.view.cutting_lines.clear()
        self.view.draw_cutting_lines()
        self.btn_cut_mode.setChecked(False)

    # --- PROJECT FILE MANAGEMENT (EXPORT / SAVE) ---
    def save_mproj(self):
        if not self.layers:
            return
            
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Сохранить проект", "", "Проекты MangaEditor (*.mproj)"
        )
        if not file_path:
            return
            
        self.save_mproj_file(file_path)
        QMessageBox.information(self, "Сохранено", "Файл проекта .mproj сохранен.")

    def save_mproj_file(self, file_path):
        self.project_path = file_path
        with zipfile.ZipFile(file_path, 'w') as zf:
            cv_imwrite_unicode("temp_orig.png", self.original_cv_image)
            zf.write("temp_orig.png", "original.png")
            os.remove("temp_orig.png")
            
            layers_data = []
            for idx, layer in enumerate(self.layers):
                if idx == 0:
                    continue
                ly_filename = f"layer_{idx}.png"
                layer.image.save(ly_filename, "PNG")
                zf.write(ly_filename, ly_filename)
                os.remove(ly_filename)
                layers_data.append({
                    "name": layer.layer_name,
                    "visible": layer.is_visible,
                    "z_val": layer.zValue(),
                    "opacity": layer.opacity()
                })
                
            texts_data = []
            for item in self.scene.items():
                if isinstance(item, TypesetTextItem):
                    texts_data.append({
                        "text": item.raw_text,
                        "x": item.x(),
                        "y": item.y(),
                        "font_family": item.font().family(),
                        "font_size": item.font().pointSize(),
                        "font_color": item.font_color.name(),
                        "outline_color": item.outline_color.name(),
                        "outline_width": item.outline_width,
                        "vertical": item.is_vertical,
                        "alignment": int(item.document().defaultTextOption().alignment()),
                        "use_gradient": item.use_gradient,
                        "gradient_color2": item.gradient_color2.name()
                    })
                    
            project_json = {
                "layers": layers_data,
                "texts": texts_data,
                "fonts": self.project_fonts
            }
            zf.writestr("project.json", json.dumps(project_json, indent=4))

    def load_mproj_file(self, file_path):
        self.scene.clear()
        self.layers.clear()
        self.undo_stack.clear()
        self.sync_history_list()
        self.project_path = file_path
        with zipfile.ZipFile(file_path, 'r') as zf:
            orig_data = zf.read("original.png")
            img_array = np.frombuffer(orig_data, dtype=np.uint8)
            self.original_cv_image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            self.on_original_image_loaded()
            
            h, w = self.original_cv_image.shape[:2]
            q_img = QImage(self.original_cv_image.data, w, h, 3*w, QImage.Format_BGR888).copy()
            
            bg = LayerItem(w, h, "Оригинал")
            bg.image = q_img
            bg.setPixmap(QPixmap.fromImage(q_img))
            self.scene.addItem(bg)
            self.layers.append(bg)
            
            proj_data = json.loads(zf.read("project.json").decode('utf-8'))
            self.project_fonts = proj_data.get("fonts", self.project_fonts)
            
            for idx, ly_info in enumerate(proj_data.get("layers", [])):
                ly_filename = f"layer_{idx+1}.png"
                ly_data = zf.read(ly_filename)
                
                layer = LayerItem(w, h, ly_info["name"])
                layer.image.loadFromData(ly_data, "PNG")
                layer.setPixmap(QPixmap.fromImage(layer.image))
                layer.setZValue(ly_info["z_val"])
                layer.setOpacity(ly_info.get("opacity", 1.0))
                self.scene.addItem(layer)
                self.layers.append(layer)
                
            for t_info in proj_data.get("texts", []):
                item = TypesetTextItem(t_info["text"])
                item.setPos(t_info["x"], t_info["y"])
                
                font = QFont(t_info["font_family"])
                font.setPointSize(t_info["font_size"])
                item.setFont(font)
                
                item.font_color = QColor(t_info["font_color"])
                item.outline_color = QColor(t_info["outline_color"])
                item.outline_width = t_info["outline_width"]
                item.is_vertical = t_info["vertical"]
                item.use_gradient = t_info.get("use_gradient", False)
                if "gradient_color2" in t_info:
                    item.gradient_color2 = QColor(t_info["gradient_color2"])
                
                item.apply_text_layout()
                
                doc = item.document()
                opt = doc.defaultTextOption()
                opt.setAlignment(Qt.Alignment(t_info["alignment"]))
                doc.setDefaultTextOption(opt)
                self.scene.addItem(item)
            
        self.scene.setSceneRect(0, 0, w, h)
        self.view.fitInView(bg, Qt.KeepAspectRatio)
        self.sync_layers_list()
        
        self.list_layers.setCurrentRow(0)
        self.select_tool("brush")
        QMessageBox.information(self, "Загружено", "Проект успешно импортирован!")

    # --- PSD IMPORT & EXPORT ---
    def import_psd(self):
        if not PSD_TOOLS_AVAILABLE:
            QMessageBox.warning(self, "Недоступно", "Библиотека psd-tools не установлена.")
            return
        file_path, _ = QFileDialog.getOpenFileName(self, "Импорт PSD", "", "Photoshop файлы (*.psd)")
        if file_path:
            self.import_psd_file(file_path)

    def import_psd_file(self, file_path):
        try:
            psd = PSDImage.open(file_path)
            self.scene.clear()
            self.layers.clear()
            self.undo_stack.clear()
            self.sync_history_list()
            merged_img = psd.composite()
            w, h = merged_img.width, merged_img.height
            open_cv_image = np.array(merged_img)
            self.original_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
            self.on_original_image_loaded()
            
            bg = LayerItem(w, h, "PSD Оригинал")
            q_img = QImage(self.original_cv_image.data, w, h, 3*w, QImage.Format_BGR888).copy()
            bg.image = q_img
            bg.setPixmap(QPixmap.fromImage(q_img))
            self.scene.addItem(bg)
            self.layers.append(bg)
            
            for idx, psd_layer in enumerate(psd):
                if psd_layer.is_group():
                    continue
                if getattr(psd_layer, 'kind', '') == 'type':
                    text_data = psd_layer.text
                    item = TypesetTextItem(text_data if text_data else "Текст")
                    item.setPos(psd_layer.left, psd_layer.top)
                    self.scene.addItem(item)
                else:
                    pil_ly = psd_layer.topil()
                    if pil_ly:
                        layer = LayerItem(w, h, psd_layer.name if psd_layer.name else f"Слой {idx}")
                        ly_q = QImage(w, h, QImage.Format_ARGB32_Premultiplied)
                        ly_q.fill(Qt.transparent)
                        
                        painter = QPainter(ly_q)
                        ly_bytes = pil_ly.tobytes()
                        q_pix = QPixmap.fromImage(QImage(ly_bytes, pil_ly.width, pil_ly.height, QImage.Format_RGBA8888).copy())
                        painter.drawPixmap(psd_layer.left, psd_layer.top, q_pix)
                        painter.end()
                        
                        layer.image = ly_q
                        layer.setPixmap(QPixmap.fromImage(ly_q))
                        self.scene.addItem(layer)
                        self.layers.append(layer)
                        
            self.scene.setSceneRect(0, 0, w, h)
            self.view.fitInView(bg, Qt.KeepAspectRatio)
            self.sync_layers_list()
            
            self.list_layers.setCurrentRow(0)
            self.select_tool("brush")
            QMessageBox.information(self, "Импорт PSD завершен", f"Загружено слоев: {len(self.layers)}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось импортировать PSD: {e}")

    def export_psd(self):
        if not PYTOSHOP_AVAILABLE or not self.layers:
            QMessageBox.warning(self, "Недоступно", "Библиотека pytoshop не установлена.")
            return
            
        file_path, _ = QFileDialog.getSaveFileName(self, "Экспорт PSD", "", "Photoshop файлы (*.psd)")
        if not file_path:
            return
            
        try:
            self.export_psd_file(file_path)
            QMessageBox.information(self, "Экспорт PSD завершен", "PSD-файл успешно сохранен со всеми слоями!")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось экспортировать PSD: {e}")

    def export_psd_file(self, file_path):
        if not PYTOSHOP_AVAILABLE or not self.layers:
            raise RuntimeError("Библиотека pytoshop или слои отсутствуют.")
            
        h, w = self.layers[0].height, self.layers[0].width
        
        psd_layers = []
        for idx, layer in enumerate(self.layers):
            q_img = layer.image.convertToFormat(QImage.Format_RGBA8888)
            width = q_img.width()
            height = q_img.height()
            bytes_per_line = q_img.bytesPerLine()
            
            ptr = q_img.constBits()
            arr = np.frombuffer(ptr, dtype=np.uint8).reshape((height, bytes_per_line // 4, 4))
            if bytes_per_line // 4 != width:
                arr = arr[:, :width, :]
            
            arr = arr.copy()
            r = arr[:, :, 0]
            g = arr[:, :, 1]
            b = arr[:, :, 2]
            a = arr[:, :, 3]
            
            channel_data = {
                0: r,
                1: g,
                2: b,
                -1: a
            }
            
            psd_ly = pytoshop.user.nested_layers.Image(
                name=layer.layer_name,
                visible=layer.is_visible,
                opacity=int(layer.opacity() * 255) if hasattr(layer, 'opacity') else 255,
                top=0,
                left=0,
                bottom=h,
                right=w,
                channels=channel_data
            )
            psd_layers.append(psd_ly)
            
        psd_file = pytoshop.user.nested_layers.nested_layers_to_psd(psd_layers, color_mode=3)
        with open(file_path, 'wb') as f:
            psd_file.write(f)

    def export_image(self):
        if not self.layers:
            return
        file_path, _ = QFileDialog.getSaveFileName(self, "Экспорт изображения", "", "Изображения (*.png *.jpg)")
        if file_path:
            scene_rect = self.scene.sceneRect()
            output = QImage(scene_rect.size().toSize(), QImage.Format_ARGB32)
            output.fill(Qt.transparent)
            
            painter = QPainter(output)
            painter.setRenderHint(QPainter.Antialiasing)
            painter.setRenderHint(QPainter.TextAntialiasing)
            self.scene.render(painter)
            painter.end()
            
            output.save(file_path)
            QMessageBox.information(self, "Экспорт", "Изображение успешно экспортировано!")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    editor = MangaEditorApp()
    editor.show()
    sys.exit(app.exec())
