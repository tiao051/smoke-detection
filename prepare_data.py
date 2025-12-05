import os
import glob
from pathlib import Path
from PIL import Image # Cần cài thư viện: pip install Pillow tqdm
from tqdm import tqdm
import sys

# --- CẤU HÌNH ---
# Đường dẫn đến folder gốc chứa dataset (nơi có các folder con 'train', 'test'...)
DATASET_ROOT = r"d:\pet-project\smoke-detection\dataset\d-fire"  # <--- SỬA LẠI ĐƯỜNG DẪN NÀY

# Các đuôi ảnh chấp nhận
IMG_FORMATS = ['.jpg', '.jpeg', '.png', '.bmp']

def check_integrity(folder_name):
    folder_path = os.path.join(DATASET_ROOT, folder_name)
    if not os.path.exists(folder_path):
        print(f"⚠️ Không tìm thấy thư mục: {folder_name} (Bỏ qua)")
        return None

    print(f"\n--- Đang kiểm tra thư mục: {folder_name} ---")
    
    # D-Fire trên Kaggle thường để ảnh và label chung 1 chỗ, hoặc chia images/labels
    # Script này sẽ tự tìm cả 2 trường hợp
    images_list = []
    for ext in IMG_FORMATS:
        images_list.extend(glob.glob(os.path.join(folder_path, '**', f'*{ext}'), recursive=True))
    
    if not images_list:
        print(f"❌ Không tìm thấy ảnh nào trong {folder_name}")
        return None

    stats = {
        'total_images': len(images_list),
        'corrupt_images': 0,
        'missing_labels': 0, # Không có file .txt
        'empty_labels': 0,   # Có file .txt nhưng rỗng (Background image chuẩn)
        'valid_objects': 0,  # Tổng số box đếm được
        'class_counts': {},  # Đếm số lượng từng class
        'errors': []
    }

    print(f"🔍 Tìm thấy {len(images_list)} ảnh. Đang quét...")

    for img_path in tqdm(images_list):
        img_path_obj = Path(img_path)
        
        # 1. Kiểm tra ảnh có mở được không
        try:
            with Image.open(img_path) as img:
                img.verify() # Check lỗi corrupt
        except Exception as e:
            stats['corrupt_images'] += 1
            stats['errors'].append(f"Ảnh lỗi: {img_path_obj.name}")
            continue

        # 2. Tìm file label tương ứng
        # Giả định label cùng tên, nằm cùng chỗ hoặc trong folder labels tương ứng
        label_path = None
        
        # Case 1: Cùng thư mục
        potential_path = img_path_obj.with_suffix('.txt')
        if potential_path.exists():
            label_path = potential_path
        
        # Case 2: Cấu trúc images/ labels/ song song
        if not label_path:
            # Thử thay 'images' bằng 'labels' trong đường dẫn
            parts = list(img_path_obj.parts)
            if 'images' in parts:
                idx = parts.index('images')
                parts[idx] = 'labels'
                potential_path_2 = Path(*parts).with_suffix('.txt')
                if potential_path_2.exists():
                    label_path = potential_path_2

        # 3. Đọc nội dung Label
        if label_path and label_path.exists():
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    
                if not lines:
                    stats['empty_labels'] += 1 # Đây là ảnh Background (Tốt)
                else:
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            
                            # Đếm class
                            stats['class_counts'][cls_id] = stats['class_counts'].get(cls_id, 0) + 1
                            stats['valid_objects'] += 1
                            
                            # Check tọa độ
                            coords = [float(x) for x in parts[1:5]]
                            if any(c < 0 or c > 1 for c in coords):
                                stats['errors'].append(f"Tọa độ sai trong file: {label_path.name}")
            except Exception as e:
                stats['errors'].append(f"Lỗi đọc label {label_path.name}: {e}")
        else:
            stats['missing_labels'] += 1 # YOLO sẽ coi là background, nhưng cần cảnh báo

    return stats

def print_report(stats, name):
    if not stats: return
    print(f"\n📊 KẾT QUẢ KIỂM TRA TẬP: {name.upper()}")
    print(f"- Tổng số ảnh:      {stats['total_images']}")
    print(f"- Ảnh bị lỗi (Corrupt): {stats['corrupt_images']} (Cần xóa ngay)")
    print(f"- File Label rỗng:  {stats['empty_labels']} (Ảnh Background - Tốt)")
    print(f"- Thiếu file Label: {stats['missing_labels']} (YOLO sẽ coi là background)")
    print(f"- Tổng số Object:   {stats['valid_objects']}")
    print(f"- Thống kê Class:   {stats['class_counts']} (Nên là 0 và 1)")
    
    if stats['errors']:
        print("\n⚠️ CÁC LỖI NGHIÊM TRỌNG TÌM THẤY:")
        for err in stats['errors'][:10]: # In 10 lỗi đầu tiên
            print(f"  - {err}")
        if len(stats['errors']) > 10: print("  ... và nhiều lỗi khác.")

def main():
    print("🚀 BẮT ĐẦU KIỂM TRA DATASET D-FIRE...")
    
    # Kiểm tra các folder phổ biến
    train_stats = check_integrity('train')
    test_stats = check_integrity('test')
    val_stats = check_integrity('val') # Có thể không có

    print_report(train_stats, 'train')
    print_report(val_stats, 'val')
    print_report(test_stats, 'test')

    print("\n✅ HOÀN TẤT.")

if __name__ == "__main__":
    main()