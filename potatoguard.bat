@echo off
cd /d E:\potato_disease_app
call env\Scripts\activate.bat
start http://localhost:5000
streamlit run app.py --server.port 5000
pause
