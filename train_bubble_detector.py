import os
from roboflow import Roboflow
from ultralytics import YOLO

def main():
    # 1. Скачивание датасета из Roboflow
    print("Инициализация Roboflow и скачивание датасета...")
    rf = Roboflow(api_key="BVKkPbjAaYzCUSde0PCn")
    project = rf.workspace("object-detection-gjmfn").project("my-first-project-yfgml")
    version = project.version(3)
    dataset = version.download("yolov8")
    
    dataset_yaml = os.path.join(dataset.location, "data.yaml")
    print(f"Датасет успешно скачан в: {dataset.location}")
    print(f"Файл конфигурации YOLO: {dataset_yaml}")
    
    # 2. Инициализация предобученной модели YOLOv8 Small
    # YOLOv8 Small — более мощная модель (11.2 млн параметров против 3.2 млн у Nano),
    # она гораздо лучше обучается сложным шрифтам и мелкому фоновому тексту.
    print("Инициализация модели YOLOv8 Small...")
    model = YOLO("yolov8s.pt")
    
    # 3. Запуск обучения
    # Мы обучаем на CPU. 150 эпох — отличный режим для полноценного обучения на всю ночь.
    print("Запуск обучения модели на CPU...")
    results = model.train(
        data=dataset_yaml,
        epochs=150,        # 150 эпох для максимальной сходимости модели
        imgsz=640,         # Размер картинок 640x640
        device="cpu",      # Обучаем на процессоре
        batch=8,           # Размер батча (8 картинок за раз)
        workers=2,         # Ограничиваем количество потоков процессора
        project="manga_bubble_train",
        name="custom_yolov8s"
    )
    
    print("Обучение завершено!")
    
    # 4. Экспорт обученной модели в ONNX
    print("Экспортируем модель в формат ONNX...")
    onnx_path = model.export(format="onnx", imgsz=640)
    print(f"Модель успешно экспортирована в ONNX: {onnx_path}")
    
    # Копируем полученную модель в папку моделей нашего проекта
    dest_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, "custom_detector.onnx")
    
    # Ищем экспортированный файл best.onnx в папке runs
    source_onnx = None
    runs_dir = os.path.join(os.path.abspath("."), "runs")
    if os.path.exists(runs_dir):
        for root, dirs, files in os.walk(runs_dir):
            if "best.onnx" in files:
                source_onnx = os.path.join(root, "best.onnx")
                break
                
    if source_onnx and os.path.exists(source_onnx):
        import shutil
        shutil.copy(source_onnx, dest_path)
        print(f"Кастомная модель успешно скопирована в: {dest_path}")
        print("Теперь мы можем переключить детекцию в приложении на вашу новую модель!")
    else:
        print(f"Внимание: Не удалось автоматически найти файл best.onnx. Пожалуйста, скопируйте его вручную в {dest_path}")

if __name__ == '__main__':
    main()
