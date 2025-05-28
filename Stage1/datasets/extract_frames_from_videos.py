import os
import shutil

def copy_videos_from_directory(video_root, output_root, ext='mp4'):
    """
    從 video_root 資料夾（包含所有子資料夾）中找到所有指定格式的影片檔案，並複製到 output_root。
    """
    if not os.path.exists(output_root):
        os.makedirs(output_root)
        
    for root, dirs, files in os.walk(video_root):
        for file in files:
            if file.endswith('.' + ext):
                video_path = os.path.join(root, file)
                
                # 取得相對於 video_root 的路徑，保留原始資料夾結構
                relative_path = os.path.relpath(root, video_root)
                target_dir = os.path.join(output_root, relative_path)
                
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                
                # 複製影片到新的資料夾
                target_path = os.path.join(target_dir, file)
                shutil.copy2(video_path, target_path)
                print(f"已複製: {video_path} -> {target_path}")

# 使用範例：
if __name__ == '__main__':
    copy_videos_from_directory('/home/video_dataset/test', '/root/video_generation/video_enhancement/GFPGAN/videos')