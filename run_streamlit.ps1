# PowerShell script to run Streamlit app
Write-Host "🚀 Starting Streamlit Web Application..." -ForegroundColor Green
Write-Host ""
Write-Host "📱 Opening browser to http://localhost:8501" -ForegroundColor Cyan
Write-Host "⏹️  Press Ctrl+C to stop the server" -ForegroundColor Yellow
Write-Host ""

# Activate virtual environment and run streamlit
& .venv\Scripts\Activate.ps1
$env:PYTHONPATH = "."
streamlit run app/streamlit_app.py --server.port 8501

