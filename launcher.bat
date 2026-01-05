@echo off
echo ===========================================
echo      Fruit & Vegetable Quality Checker
echo ===========================================
echo.
echo Select an option:
echo 1. Build Windows Executable (.exe)
echo 2. Run Web Application (Streamlit)
echo 3. Run Desktop GUI (Python)
echo.
set /p choice="Enter your choice (1-3): "

if "%choice%"=="1" goto build_exe
if "%choice%"=="2" goto run_web
if "%choice%"=="3" goto run_gui
goto end

:build_exe
echo.
echo Building executable... this may take a minute.
echo Ensuring PyInstaller is installed...
pip install pyinstaller
echo.
echo Running PyInstaller...
pyinstaller --noconfirm --onefile --windowed --name "FruitQualityChecker" --add-data "fruits_veg_model.h5;." --add-data "scripts;scripts" gui_app.py
echo.
echo Build complete!
echo You can find your app in the 'dist' folder: dist\FruitQualityChecker.exe
pause
goto end

:run_web
echo.
echo Starting Web App...
streamlit run web_app.py
goto end

:run_gui
echo.
echo Starting Desktop GUI...
python gui_app.py
goto end

:end
