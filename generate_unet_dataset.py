# -*- coding: utf-8 -*-
import os
import random
import urllib.request
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import io

def download_fonts():
    font_dir = 'fonts'
    os.makedirs(font_dir, exist_ok=True)
    
    # Ссылка на японский шрифт Kosugi Maru и Noto Sans JP
    fonts = {
        'KosugiMaru-Regular.ttf': 'https://github.com/google/fonts/raw/main/ofl/kosugimaru/KosugiMaru-Regular.ttf',
        'NotoSansJP-Regular.ttf': 'https://github.com/google/fonts/raw/main/ofl/notosansjp/NotoSansJP-Regular.ttf'
    }
    
    downloaded = []
    for name, url in fonts.items():
        path = os.path.join(font_dir, name)
        if not os.path.exists(path):
            print(f'Скачивание японского шрифта {name}...')
            try:
                urllib.request.urlretrieve(url, path)
                downloaded.append(path)
            except Exception as e:
                print(f'Не удалось скачать {name}: {e}')
        else:
            downloaded.append(path)
            
    # Добавляем стандартные русские/английские шрифты Windows
    sys_fonts = [
        'C:\\Windows\\Fonts\\arial.ttf',
        'C:\\Windows\\Fonts\\times.ttf',
        'C:\\Windows\\Fonts\\msgothic.ttc',
        'C:\\Windows\\Fonts\\meiryo.ttc'
    ]
    for sf in sys_fonts:
        if os.path.exists(sf):
            downloaded.append(sf)
            
    return downloaded

def harvest_real_backgrounds():
    """Извлекает участки без текста из оригинального YOLO датасета в папке education"""
    bg_patches = []
    images_dir = 'education/train/images'
    labels_dir = 'education/train/labels'
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        print('Папка с YOLO-датасетом (education) не найдена. Будем использовать только математические фоны.')
        return bg_patches
        
    print('Сбор реальных фонов из папки education...')
    img_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_name in img_files:
        img_path = os.path.join(images_dir, img_name)
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(labels_dir, base_name + '.txt')
        
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Создаем маску текстовых областей (True - текст, False - пусто)
        text_mask = np.zeros((h, w), dtype=bool)
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, x_c, y_c, wb, hb = map(float, parts[:5])
                        # Переводим в пиксельные координаты
                        x1 = int((x_c - wb/2) * w) - 15
                        y1 = int((y_c - hb/2) * h) - 15
                        x2 = int((x_c + wb/2) * w) + 15
                        y2 = int((y_c + hb/2) * h) + 15
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(w, x2)
                        y2 = min(h, y2)
                        text_mask[y1:y2, x1:x2] = True
                        
        # Нарезаем случайные патчи 256x256, не содержащие текст
        for _ in range(50): # до 50 попыток на изображение
            px = random.randint(0, w - 256)
            py = random.randint(0, h - 256)
            # Проверяем, есть ли текст в этом патче
            if not np.any(text_mask[py:py+256, px:px+256]):
                patch = img[py:py+256, px:px+256].copy()
                bg_patches.append(patch)
            if len(bg_patches) >= 600: # лимит 600 реальных патчей
                break
        if len(bg_patches) >= 600:
            break
                    
    print(f'Собрано реальных фонов: {len(bg_patches)}')
    return bg_patches

def generate_screentone(w=256, h=256):
    """Генерирует математический манга-скринтон"""
    x = np.arange(w)
    y = np.arange(h)
    X, Y = np.meshgrid(x, y)
    
    # Поворачиваем координаты на 45 градусов
    angle = np.pi / 4
    X_rot = X * np.cos(angle) - Y * np.sin(angle)
    Y_rot = X * np.sin(angle) + Y * np.cos(angle)
    
    frequency = random.uniform(0.08, 0.25)
    pattern = (np.sin(X_rot * frequency) + np.sin(Y_rot * frequency)) / 2.0
    
    thresh = random.uniform(-0.6, 0.4)
    screentone = np.where(pattern > thresh, 255, random.randint(180, 230)).astype(np.uint8)
    
    # Переводим в BGR
    screentone_bgr = cv2.cvtColor(screentone, cv2.COLOR_GRAY2BGR)
    return screentone_bgr

def generate_speedlines(w=256, h=256):
    """Генерирует спидлайны"""
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    step = random.randint(8, 25)
    offset = random.randint(-40, 40)
    for x in range(-50, w + 50, step):
        w_line = random.randint(1, 3)
        col = random.randint(0, 100)
        cv2.line(img, (x, 0), (x + offset, h), (col, col, col), w_line)
    return img

def generate_gradient(w=256, h=256):
    """Генерирует линейный градиент"""
    start = random.randint(150, 240)
    end = random.randint(80, start - 30)
    grad = np.linspace(start, end, w).astype(np.uint8)
    grad = np.tile(grad, (h, 1))
    if random.choice([True, False]):
        grad = grad.T
    return cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)

def generate_solid(w=256, h=256):
    """Генерирует однородный цвет с небольшим шумом"""
    gray = random.randint(120, 255)
    img = np.ones((h, w, 3), dtype=np.uint8) * gray
    noise = np.random.normal(0, random.uniform(1.0, 5.0), img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img

def get_random_text():
    japanese_chars = 'あいうえおかきくけこさしすせそたちつてとなにぬねのはひふへほまみむめもやゆよらりるれろわをんアイウエオカキクケコサシスセソタチツテトナニヌネノハヒфхехомамяюя夢幻闘魔境界戦線殺生丸犬夜叉'
    russian_chars = 'АБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЭЮЯабвгдежзийклмнопрстуфхцчшщэюя'
    english_chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz!?.1234567890'
    
    lang = random.choices(['jp', 'ru', 'en'], weights=[0.5, 0.3, 0.2])[0]
    if lang == 'jp':
        return ''.join(random.choices(japanese_chars, k=random.randint(2, 6)))
    elif lang == 'ru':
        return ''.join(random.choices(russian_chars, k=random.randint(3, 8)))
    else:
        return ''.join(random.choices(english_chars, k=random.randint(3, 8)))

def build_dataset(num_samples=2000):
    print(f'Начало генерации датасета ({num_samples} образцов)...')
    
    fonts = download_fonts()
    if not fonts:
        print('Критическая ошибка: Шрифты не найдены!')
        return
        
    real_bgs = harvest_real_backgrounds()
    
    out_dir = 'unet_dataset'
    os.makedirs(os.path.join(out_dir, 'train', 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'train', 'masks'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'val', 'images'), exist_ok=True)
    os.makedirs(os.path.join(out_dir, 'val', 'masks'), exist_ok=True)
    
    for idx in range(num_samples):
        # 1. Выбираем фон
        bg_type = random.choices(['real', 'screentone', 'speedlines', 'gradient', 'solid'], weights=[0.4, 0.25, 0.1, 0.15, 0.1])[0]
        if bg_type == 'real' and real_bgs:
            bg = random.choice(real_bgs).copy()
        elif bg_type == 'screentone':
            bg = generate_screentone()
        elif bg_type == 'speedlines':
            bg = generate_speedlines()
        elif bg_type == 'gradient':
            bg = generate_gradient()
        else:
            bg = generate_solid()
            
        bg_pil = Image.fromarray(cv2.cvtColor(bg, cv2.COLOR_BGR2RGB))
        w_img, h_img = bg_pil.size
        
        mask_pil = Image.new('L', (w_img, h_img), 0)
        draw_txt = ImageDraw.Draw(bg_pil)
        draw_mask = ImageDraw.Draw(mask_pil)
        
        num_lines = random.randint(1, 2)
        for _ in range(num_lines):
            text = get_random_text()
            font_path = random.choice(fonts)
            font_size = random.randint(20, 48)
            
            try:
                font = ImageFont.truetype(font_path, font_size)
            except Exception:
                continue
                
            tx = random.randint(10, w_img - 150)
            ty = random.randint(10, h_img - 60)
            
            style = random.choice(['white_black', 'black_white', 'pure_black', 'pure_white'])
            if style == 'white_black':
                t_color = (255, 255, 255)
                o_color = (0, 0, 0)
                stroke_w = random.randint(2, 5)
            elif style == 'black_white':
                t_color = (0, 0, 0)
                o_color = (255, 255, 255)
                stroke_w = random.randint(2, 5)
            elif style == 'pure_black':
                t_color = (0, 0, 0)
                o_color = None
                stroke_w = 0
            else:
                t_color = (255, 255, 255)
                o_color = None
                stroke_w = 0
                
            draw_txt.text((tx, ty), text, font=font, fill=t_color, stroke_width=stroke_w, stroke_fill=o_color)
            draw_mask.text((tx, ty), text, font=font, fill=255, stroke_width=stroke_w, stroke_fill=255)
            
        buffer = io.BytesIO()
        bg_pil.save(buffer, format='JPEG', quality=random.randint(45, 85))
        buffer.seek(0)
        final_img = Image.open(buffer)
        
        split = 'val' if idx % 10 == 0 else 'train'
        
        img_name = f'sample_{idx:05d}.jpg'
        mask_name = f'sample_{idx:05d}.png'
        
        final_img.save(os.path.join(out_dir, split, 'images', img_name))
        mask_pil.save(os.path.join(out_dir, split, 'masks', mask_name))
        
        if (idx + 1) % 200 == 0:
            print(f'Сгенерировано {idx + 1}/{num_samples}...')
            
    print(f'Датасет успешно сгенерирован и сохранен в папку {out_dir}!')

if __name__ == '__main__':
    build_dataset(2000)
