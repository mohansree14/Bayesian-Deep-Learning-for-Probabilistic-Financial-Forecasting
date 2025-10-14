#!/usr/bin/env python
"""
Generate professional table image for HPO Search Space.
Creates a high-quality PNG image of the hyperparameter optimization configuration table.
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
import numpy as np
from pathlib import Path


def create_hpo_table_image():
    """Generate professional HPO search space table as image."""
    print("[*] Generating HPO Search Space Table Image...")
    
    # Table data
    headers = [
        'Hyperparameter',
        'Type',
        'Min Value',
        'Max Value',
        'Step Size',
        'Possible Values',
        'Distribution',
        'Configs'
    ]
    
    data = [
        ['Hidden Size', 'Integer', '64', '256', '64', '64, 128, 192, 256', 'Uniform', '4'],
        ['Num Layers', 'Integer', '1', '3', '1', '1, 2, 3', 'Uniform', '3'],
        ['Dropout', 'Float', '0.0', '0.4', 'Continuous', '0.0–0.4', 'Uniform', '∞'],
        ['Learning Rate', 'Float', '0.0001', '0.005', 'Continuous', '1e-4 to 5e-3', 'Log-Uniform', '∞'],
        ['Batch Size', 'Categorical', '—', '—', '—', '32, 64, 128', 'Discrete', '3'],
    ]
    
    # Create figure - adjusted for table only
    fig, ax = plt.subplots(figsize=(16, 4.5))
    ax.axis('off')
    
    # Table dimensions
    n_rows = len(data) + 1  # +1 for header
    n_cols = len(headers)
    
    # Cell dimensions
    cell_height = 0.14
    cell_width = 1.0 / n_cols
    
    # Starting position - adjusted to use full space
    start_y = 0.88
    start_x = 0.01
    
    # Colors
    header_color = '#1E40AF'  # Dark blue
    row_colors = ['#F3F4F6', '#FFFFFF']  # Alternating gray and white
    text_color = '#1F2937'  # Dark gray
    border_color = '#D1D5DB'  # Light gray
    
    # Draw header
    for col, header in enumerate(headers):
        x = start_x + col * cell_width
        y = start_y
        
        # Header cell background
        rect = Rectangle((x, y), cell_width, cell_height,
                         linewidth=1.5, edgecolor=border_color,
                         facecolor=header_color, transform=ax.transAxes)
        ax.add_patch(rect)
        
        # Header text (white)
        ax.text(x + cell_width/2, y + cell_height/2, header,
               ha='center', va='center', fontsize=11, fontweight='bold',
               color='white', transform=ax.transAxes)
    
    # Draw data rows
    for row_idx, row_data in enumerate(data):
        y = start_y - (row_idx + 1) * cell_height
        row_color = row_colors[row_idx % 2]
        
        for col_idx, cell_value in enumerate(row_data):
            x = start_x + col_idx * cell_width
            
            # Cell background
            rect = Rectangle((x, y), cell_width, cell_height,
                           linewidth=1, edgecolor=border_color,
                           facecolor=row_color, transform=ax.transAxes)
            ax.add_patch(rect)
            
            # Cell text
            fontsize = 10
            fontweight = 'bold' if col_idx == 0 else 'normal'
            
            # Adjust text wrapping for long values
            if col_idx == 5:  # Possible Values column
                fontsize = 9
            
            ax.text(x + cell_width/2, y + cell_height/2, cell_value,
                   ha='center', va='center', fontsize=fontsize,
                   fontweight=fontweight, color=text_color,
                   transform=ax.transAxes)
    
    # No footer - clean table only
    
    # Save figure
    output_path = Path('reports/figures')
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as PNG
    save_path_png = output_path / 'hpo_search_space_table.png'
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"[OK] Table image saved to: {save_path_png}")
    
    # Save as PDF
    save_path_pdf = output_path / 'hpo_search_space_table.pdf'
    plt.savefig(save_path_pdf, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"[OK] Table image (PDF) saved to: {save_path_pdf}")
    
    plt.close()
    
    print("\n[*] Table Generation Complete!")
    print(f"    PNG: {save_path_png} (300 DPI)")
    print(f"    PDF: {save_path_pdf} (Vector)")


if __name__ == "__main__":
    create_hpo_table_image()
    print("\n[*] Done! Table image ready for use in reports and presentations.")
