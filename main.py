import os

print("Welcome to the Fruit and Vegetable Quality Checker!")
print("1. Upload and Classify Image")
print("2. Real-Time Camera Classification")
choice = input("Enter your choice (1/2): ")

if choice == '1':
    os.system("python scripts/predict_image.py")
elif choice == '2':
    os.system("python scripts/predict_camera.py")
else:
    print("Invalid choice. Exiting.")
