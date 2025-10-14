#!/usr/bin/env python
"""
Generate professional table image for Technical Indicators.
Creates a high-quality PNG/PDF image of all 21 technical indicators.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from pathlib import Path


def create_indicators_table_image():
    """Generate professional technical indicators table as image."""
    print("[*] Generating Technical Indicators Table Image...")
    
    # Table data - all 21 indicators
    headers = [
        '#',
        'Indicator Name',
        'Category',
        'Formula',
        'Parameters',
        'Purpose'
    ]
    
    data = [
        ['1', 'Open', 'Price', 'Raw OHLC', '—', 'Opening price'],
        ['2', 'High', 'Price', 'Raw OHLC', '—', 'Highest price'],
        ['3', 'Low', 'Price', 'Raw OHLC', '—', 'Lowest price'],
        ['4', 'Close', 'Price', 'Raw OHLC', '—', 'Closing price'],
        ['5', 'Adj Close', 'Price', 'Close × Adj Factor', '—', 'Adjusted price'],
        ['6', 'Volume', 'Volume', 'Trading volume', '—', 'Market activity'],
        ['7', 'Return (1d)', 'Returns', '(Cₜ - Cₜ₋₁) / Cₜ₋₁', 'window=1', 'Daily % change'],
        ['8', 'Log Return', 'Returns', 'ln(Cₜ / Cₜ₋₁)', 'window=1', 'Log return'],
        ['9', 'SMA-10', 'Trend', 'Avg(Close, 10)', 'window=10', 'Short trend'],
        ['10', 'SMA-20', 'Trend', 'Avg(Close, 20)', 'window=20', 'Medium trend'],
        ['11', 'EMA-12', 'Trend', 'EMA(Close, 12)', 'span=12', 'Fast trend'],
        ['12', 'EMA-26', 'Trend', 'EMA(Close, 26)', 'span=26', 'Slow trend'],
        ['13', 'RSI-14', 'Momentum', '100 - 100/(1 + RS)', 'window=14', 'Overbought/Oversold'],
        ['14', 'MACD', 'Momentum', 'EMA(12) - EMA(26)', 'fast=12, slow=26', 'Trend strength'],
        ['15', 'MACD Signal', 'Momentum', 'EMA(MACD, 9)', 'signal=9', 'Buy/sell signal'],
        ['16', 'MACD Hist', 'Momentum', 'MACD - Signal', '—', 'Momentum change'],
        ['17', 'BB Middle', 'Volatility', 'SMA(Close, 20)', 'window=20', 'Mean reversion'],
        ['18', 'BB Upper', 'Volatility', 'BB_mid + 2σ', 'std=2', 'Upper boundary'],
        ['19', 'BB Lower', 'Volatility', 'BB_mid - 2σ', 'std=2', 'Lower boundary'],
        ['20', 'Stoch %K', 'Momentum', '100(C-Lₙ)/(Hₙ-Lₙ)', 'window=14', 'Price position'],
        ['21', 'Stoch %D', 'Momentum', 'SMA(%K, 3)', 'window=3', 'Signal line'],
    ]
    
    # Create figure - compact size for table only
    fig, ax = plt.subplots(figsize=(18, 11))
    ax.axis('off')
    
    # Table dimensions
    n_rows = len(data) + 1  # +1 for header
    n_cols = len(headers)
    
    # Cell dimensions
    cell_height = 0.042
    cell_width = 1.0 / n_cols
    
    # Column widths (custom)
    col_widths = [0.04, 0.16, 0.12, 0.20, 0.16, 0.32]  # Custom widths
    
    # Starting position - adjusted for table only
    start_y = 0.96
    start_x = 0.01
    
    # Colors
    header_color = '#1E40AF'
    category_colors = {
        'Price': '#DBEAFE',
        'Volume': '#FEF3C7',
        'Returns': '#D1FAE5',
        'Trend': '#FCE7F3',
        'Momentum': '#FED7AA',
        'Volatility': '#E0E7FF'
    }
    text_color = '#1F2937'
    border_color = '#D1D5DB'
    
    # Draw header
    x_pos = start_x
    for col_idx, header in enumerate(headers):
        width = col_widths[col_idx]
        
        # Header cell
        rect = Rectangle((x_pos, start_y), width, cell_height,
                         linewidth=1.5, edgecolor=border_color,
                         facecolor=header_color, transform=ax.transAxes)
        ax.add_patch(rect)
        
        # Header text
        ax.text(x_pos + width/2, start_y + cell_height/2, header,
               ha='center', va='center', fontsize=10, fontweight='bold',
               color='white', transform=ax.transAxes)
        
        x_pos += width
    
    # Draw data rows
    for row_idx, row_data in enumerate(data):
        y = start_y - (row_idx + 1) * cell_height
        category = row_data[2]  # Category column
        row_color = category_colors.get(category, '#FFFFFF')
        
        x_pos = start_x
        for col_idx, cell_value in enumerate(row_data):
            width = col_widths[col_idx]
            
            # Cell background
            rect = Rectangle((x_pos, y), width, cell_height,
                           linewidth=0.8, edgecolor=border_color,
                           facecolor=row_color, transform=ax.transAxes)
            ax.add_patch(rect)
            
            # Cell text
            fontsize = 9
            fontweight = 'bold' if col_idx == 0 else 'normal'
            
            # Center text for first 3 columns, left-align for rest
            ha = 'center' if col_idx < 3 else 'left'
            x_text = x_pos + width/2 if col_idx < 3 else x_pos + 0.01
            
            ax.text(x_text, y + cell_height/2, cell_value,
                   ha=ha, va='center', fontsize=fontsize,
                   fontweight=fontweight, color=text_color,
                   transform=ax.transAxes)
            
            x_pos += width
    
    # Save figure
    output_path = Path('reports/figures')
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as PNG
    save_path_png = output_path / 'technical_indicators_table.png'
    plt.savefig(save_path_png, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"[OK] Table image saved to: {save_path_png}")
    
    # Save as PDF
    save_path_pdf = output_path / 'technical_indicators_table.pdf'
    plt.savefig(save_path_pdf, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print(f"[OK] Table image (PDF) saved to: {save_path_pdf}")
    
    plt.close()
    
    print("\n[*] Table Generation Complete!")
    print(f"    PNG: {save_path_png} (300 DPI)")
    print(f"    PDF: {save_path_pdf} (Vector)")


if __name__ == "__main__":
    create_indicators_table_image()
    print("\n[*] Done! Technical indicators table ready for reports.")
