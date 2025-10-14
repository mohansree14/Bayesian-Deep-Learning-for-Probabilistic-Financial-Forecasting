@echo off
echo Starting Streamlit Web Application...
echo.
echo Opening browser to http://localhost:8501
echo Press Ctrl+C to stop the server
echo.

REM Activate virtual environment and run streamlit
call .venv\Scripts\activate.bat
set PYTHONPATH=.
streamlit run app/streamlit_app.py --server.port 8501

pause

