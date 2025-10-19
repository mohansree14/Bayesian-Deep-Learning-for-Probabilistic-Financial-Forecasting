#!/usr/bin/env python3
"""
Streamlit app launcher with proper Python path setup.
This script ensures the src modules can be imported correctly on Streamlit Cloud.
"""

import sys
import os
from pathlib import Path

# Setup paths
current_dir = Path(__file__).parent
project_root = current_dir.parent

# Add paths to sys.path
paths_to_add = [
    str(project_root),
    str(current_dir),
    "/mount/src/bayesian-deep-learning-for-probabilistic-financial-forecasting",  # Streamlit Cloud path
]

for path in paths_to_add:
    if path not in sys.path:
        sys.path.insert(0, path)

# Now import and run the actual app
if __name__ == "__main__":
    # Import the main app module
    from streamlit_app import main
    main()