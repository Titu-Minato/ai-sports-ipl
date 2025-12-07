@echo off
cd /d "C:\Users\parma\OneDrive\Desktop\AI Sports"
call "venv\Scripts\activate.bat"
python app.py
start http://127.0.0.1:5000
pause
