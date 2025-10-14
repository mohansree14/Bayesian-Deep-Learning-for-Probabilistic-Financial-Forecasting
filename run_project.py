#!/usr/bin/env python
"""
Complete project runner - executes the entire ML pipeline and starts web app.
"""
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def run_command(cmd, description, wait=True):
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        if wait:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            print("✅ SUCCESS")
            if result.stdout:
                print("Output:", result.stdout)
            return True
        else:
            # Run in background
            subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("✅ STARTED (running in background)")
            return True
    except subprocess.CalledProcessError as e:
        print("❌ FAILED")
        print("Error:", e.stderr)
        return False

def main():
    print("🎯 COMPLETE ML PROJECT EXECUTION")
    print("=" * 60)
    print("This will run the entire ML pipeline and start the web app")
    print("=" * 60)
    
    # Set environment
    import os
    os.environ['PYTHONPATH'] = '.'
    
    success_count = 0
    total_steps = 0
    
    # Step 1: Run unit tests
    total_steps += 1
    if run_command([sys.executable, "-m", "pytest", "tests/", "-v"], "Running Unit Tests"):
        success_count += 1
    
    # Step 2: Generate complete results
    total_steps += 1
    if run_command([sys.executable, "generate_results.py", "--config", "configs/lstm_baseline.yaml"], 
                  "Generating Complete Results"):
        success_count += 1
    
    # Step 3: Start Streamlit app
    total_steps += 1
    if run_command([sys.executable, "-m", "streamlit", "run", "app/streamlit_app.py", 
                   "--server.port", "8501", "--server.headless", "true"], 
                  "Starting Streamlit Web Application", wait=False):
        success_count += 1
        
        # Wait for app to start
        print("\n⏳ Waiting for Streamlit app to start...")
        time.sleep(5)
        
        # Open browser
        try:
            webbrowser.open("http://localhost:8501")
            print("🌐 Browser opened to http://localhost:8501")
        except Exception as e:
            print(f"⚠️  Could not open browser automatically: {e}")
            print("🌐 Please manually open http://localhost:8501 in your browser")
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 PROJECT EXECUTION SUMMARY")
    print('='*60)
    print(f"✅ Successful steps: {success_count}/{total_steps}")
    
    if success_count == total_steps:
        print("🎉 ALL SYSTEMS OPERATIONAL!")
        print("\n📱 Web Application: http://localhost:8501")
        print("📊 Results: Check the 'results/' folder")
        print("📝 Logs: Check the 'results/logs/' folder")
        print("\n🛑 To stop the web app: Press Ctrl+C in the terminal")
    else:
        print("⚠️  Some steps failed. Check the output above for details.")
        return 1
    
    # Keep the script running to maintain the web app
    if success_count == total_steps:
        print("\n🔄 Web application is running... Press Ctrl+C to stop")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutting down web application...")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

