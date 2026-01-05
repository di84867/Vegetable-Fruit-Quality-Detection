This project leverages a regular camera device to perform real-time quality detection of fruits and vegetables. The system captures live image or video data and applies advanced image processing and machine learning techniques such as Convolutional Neural Networks (CNN) to automatically identify freshness, defects, and spoilage indicators.

Core features include:

- Continuous, real-time monitoring of produce quality
- Automatic classification of fruits and vegetables based on color, texture, and visible defects.

## Installation & Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/YourUsername/vegetables-fruits-quality-detection.git
   cd vegetables-fruits-quality-detection
   ```

2. **Create a Virtual Environment (Optional but Recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

**To run the Desktop GUI Application:**

```bash
python gui_app.py
```

**To run the Command Line Interface:**

```bash
python main.py
```

**To run the Web Application:**

```bash
streamlit run web_app.py
```

## Building the Windows Executable

To create a standalone `.exe` file that runs without installing Python:

1.  Double-click `launcher.bat` and select **Option 1**.
2.  Or run manually:
    ```bash

    ```
3.  The executable will be in the `dist/` folder.

## Author

**Divyansh Singh**

- [LinkedIn](https://www.linkedin.com/in/divyansh-singh-26a95a248/)
- [GitHub](https://github.com/di84867)
