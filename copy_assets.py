import shutil
import os

source = r"C:\Users\singh\.gemini\antigravity\brain\5b66c430-11f0-4731-b755-0f2924d916c0\uploaded_image_1767584435067.jpg"
dest_dir = os.path.join(os.getcwd(), "assets")
dest_file = os.path.join(dest_dir, "profile_pic.jpg")

if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
    print(f"Created directory: {dest_dir}")

if os.path.exists(source):
    try:
        shutil.copy2(source, dest_file)
        print(f"Successfully copied to {dest_file}")
    except Exception as e:
        print(f"Error copying file: {e}")
else:
    print(f"Source file not found at: {source}")
