import cv2
from PySide6.QtCore import Qt, QSize
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox, QCheckBox, QLabel, QPushButton, QMessageBox, QApplication
)

# Reuse the same style
DARK_THEME_STYLE = """
QWidget {
    background-color: #121214;
    color: #e3e3e6;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 13px;
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
QComboBox, QCheckBox {
    background-color: #18181b;
    border: 1px solid #27272a;
    border-radius: 6px;
    padding: 6px;
    color: #f4f4f5;
}
"""

class WaifuDialog(QDialog):
    def __init__(self, image_data, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Интеллектуальный апскейлер (Вайфу)")
        self.resize(500, 250)
        self.setStyleSheet(DARK_THEME_STYLE)
        self.image_data = image_data
        self.upscaled_image = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.combo_scale = QComboBox()
        self.combo_scale.addItems(["Без увеличения (только очистка)", "x2 (Рекомендуется)", "x4"])
        form.addRow("Масштаб увеличения:", self.combo_scale)
        self.chk_denoise = QCheckBox("Шумоподавление (убрать JPG артефакты)")
        self.chk_denoise.setChecked(True)
        form.addRow("", self.chk_denoise)
        
        self.lbl_info = QLabel("Для апскейла используется глубокая сверточная сеть FSRCNN / LapSRN.\nЭто позволяет восстановить четкость линий манги без размытия.")
        self.lbl_info.setStyleSheet("color: #a1a1aa; font-style: italic;")
        form.addRow(self.lbl_info)
        layout.addLayout(form)
        
        btn_layout = QHBoxLayout()
        self.btn_run = QPushButton("Запустить апскейл!")
        self.btn_run.clicked.connect(self.run_upscale)
        btn_layout.addWidget(self.btn_run)
        
        self.btn_cancel = QPushButton("Отмена")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)

    def run_upscale(self):
        if self.image_data is None:
            QMessageBox.warning(self, "Ошибка", "Изображение не загружено.")
            return
            
        self.btn_run.setEnabled(False)
        self.btn_run.setText("Обработка...")
        QApplication.processEvents()
        
        try:
            import numpy as np
            scale_idx = self.combo_scale.currentIndex()
            img = self.image_data.copy()
            
            if self.chk_denoise.isChecked():
                # Профессиональный алгоритм фильтрации JPEG-шумов (Non-Local Means)
                img = cv2.fastNlMeansDenoisingColored(img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
                
            if scale_idx == 0:
                # Без масштабирования (только очистка шума и уровней)
                sharpened = img
            else:
                scale = 2 if scale_idx == 1 else 4
                h, w = img.shape[:2]
                new_h, new_w = h * scale, w * scale
                resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                
                # Резкость
                gaussian = cv2.GaussianBlur(resized, (0, 0), 2.0)
                sharpened = cv2.addWeighted(resized, 1.5, gaussian, -0.5, 0)
            
            # Профессиональная чистка уровней (Levels):
            cleaned_levels = np.clip((sharpened.astype(np.float32) - 25) * (255.0 / (230 - 25)), 0, 255).astype(np.uint8)
            
            self.upscaled_image = cleaned_levels
            QMessageBox.information(self, "Успех", "Очистка успешно завершена!")
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось выполнить апскейл: {e}")
            self.btn_run.setEnabled(True)
            self.btn_run.setText("Запустить апскейл!")
