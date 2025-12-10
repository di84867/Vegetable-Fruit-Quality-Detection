import tkinter as tk
from tkinter import font as tkFont, Label, Frame, Button, filedialog
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image, ImageTk
import cv2
import numpy as np

# Import model utilities and classification functions
from scripts.model_utils import load_model
from scripts.prediction_utils import get_prediction_details

class QualityCheckerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Fruit and Vegetable Quality Checker")
        self.root.geometry("800x700")
        self.root.configure(bg="#F0F0F0")

        self.CONFIDENCE_THRESHOLD = 90.0 # High confidence required for the top prediction.
        self.REJECTION_THRESHOLD = 10.0  # Reject if any other class has more than this confidence.
        # --- State Variables ---
        self.camera_active = False
        self.cap = None

        # --- Fonts ---
        self.title_font = tkFont.Font(family="Helvetica", size=16, weight="bold")
        self.button_font = tkFont.Font(family="Helvetica", size=11)
        self.result_font = tkFont.Font(family="Helvetica", size=12, weight="bold")

        # --- Widgets ---
        self.setup_widgets()

        self.model = load_model() # Load model once
        # --- Window close handler ---
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_widgets(self):
        # Header
        Label(self.root, text="Fruit & Vegetable Quality Checker", font=self.title_font, bg="#F0F0F0", fg="#333").pack(pady=(20, 10))

        self.video_frame = Frame(self.root, bg="#000000", bd=2, relief="sunken", width=640, height=480)
        self.video_frame.pack(pady=10)
        self.video_frame.pack_propagate(False)
        self.video_label = Label(self.video_frame, bg="#000000")
        self.video_label.pack(expand=True)

        # Result Label
        self.result_label = Label(self.root, text="Select a mode to begin", font=self.result_font, bg="#F0F0F0", fg="#555")
        self.result_label.pack(pady=(5, 15))

        # Buttons
        button_frame = Frame(self.root, bg="#F0F0F0")
        button_frame.pack(pady=10, fill="x", expand=True)

        self.camera_button = Button(button_frame, text="Start Real-Time Check", font=self.button_font, command=self.toggle_camera, bg="#2196F3", fg="white", relief="flat", padx=10, pady=5)
        self.camera_button.pack(side="left", expand=True, padx=10)

        self.upload_button = Button(button_frame, text="Upload and Classify Image", font=self.button_font, command=self.upload_and_classify, bg="#4CAF50", fg="white", relief="flat", padx=10, pady=5)
        self.upload_button.pack(side="right", expand=True, padx=10)

    def toggle_camera(self):
        if self.camera_active:
            # --- Stop Camera ---
            self.camera_active = False
            self.camera_button.config(text="Start Real-Time Check")
            if self.cap and self.cap.isOpened():
                self.cap.release()
            # Clear the label
                self.video_label.config(image='')
                self.video_label.image = None
                self.result_label.config(text="Select a mode to begin")
        else:
            # --- Start Camera ---
            self.camera_active = True
            self.camera_button.config(text="Stop Real-Time Check")
            self.cap = cv2.VideoCapture(0)
            self.result_label.config(text="Please show a vegetable or fruit.", fg="#FFA500")
            if not self.cap or not self.cap.isOpened():
                self.result_label.config(text="Error: Could not open camera.", fg="#FF0000")
                self.camera_active = False
                self.camera_button.config(text="Start Real-Time Check")
                return
            self.update_camera_feed()

    def update_camera_feed(self):
        if not self.camera_active:
            return

        ret, frame = self.cap.read()
        if ret:
            # --- Classification Logic ---
            img_for_model = cv2.resize(frame, (224, 224))
            img_array = np.expand_dims(img_for_model, axis=0) / 255.0

            # --- Display Logic ---
            display_text = ""
            text_color = (0, 255, 0) # Default to green for good prediction

            is_valid_prediction, details = get_prediction_details(
                self.model, img_array, self.CONFIDENCE_THRESHOLD, self.REJECTION_THRESHOLD
            )

            if is_valid_prediction:
                # Group results to avoid fruit/veg confusion
                display_text = f"{details['label']} ({details['confidence']:.2f}%)"
                result_color = "#008000" if "Fresh" in details['label'] else "#E53935" # Green for Fresh, Red for Rotten
                self.result_label.config(text=f"Prediction: {details['label']}\nConfidence: {details['confidence']:.2f}%", fg=result_color)
            else:
                display_text = "Please show a vegetable or fruit"
                text_color = (0, 165, 255) # Orange for awaiting status
                self.result_label.config(text="Please show a vegetable or fruit.", fg="#FFA500")

            # Draw text on the frame
            cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, text_color, 2, cv2.LINE_AA)
            
            # Convert for tkinter
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)

            frame_width = self.video_frame.winfo_width()
            frame_height = self.video_frame.winfo_height()
            img.thumbnail((frame_width, frame_height), Image.LANCZOS)

            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
        else:
            # If camera read fails, stop the process
            self.toggle_camera()
        
        # Schedule next update
        self.root.after(20, self.update_camera_feed)

    def upload_and_classify(self):
        # Stop camera if it's running
        if self.camera_active:
            self.toggle_camera()
        
        filepath = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not filepath:
            return

        try:
            # Display the selected image
            img = Image.open(filepath)
            frame_width = self.video_frame.winfo_width()
            frame_height = self.video_frame.winfo_height()
            img.thumbnail((frame_width, frame_height), Image.LANCZOS)
            photo = ImageTk.PhotoImage(img)
            self.video_label.config(image=photo)
            self.video_label.image = photo # Keep a reference

            # Get prediction
            self.result_label.config(text="Classifying...", fg="#FFA500")
            self.root.update_idletasks()
            
            img_for_model = load_img(filepath, target_size=(224, 224))
            img_array = img_to_array(img_for_model) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            is_valid_prediction, details = get_prediction_details(
                 self.model, img_array, self.CONFIDENCE_THRESHOLD, self.REJECTION_THRESHOLD
            )

            if is_valid_prediction:
                result_color = "#008000" if "Fresh" in details['label'] else "#E53935"
                result_text = f"Prediction: {details['label']}\nConfidence: {details['confidence']:.2f}%"
                self.result_label.config(text=result_text, fg=result_color)
            else:
                self.result_label.config(text="Please use an image of a single fruit or vegetable.", fg="#FFA500")

        except Exception as e:
            self.result_label.config(text=f"Error: {e}", fg="#FF0000")

    def on_closing(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = QualityCheckerApp(root)
    root.mainloop()