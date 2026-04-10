import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

base = Path(r"C:\Users\nived\OneDrive\Documents\Desktop\ds stock")
input_path = base / "random_stock_market_dataset.csv"
output_clean = base / "cleaned_stock_dataset.csv"
plot_path = base / "stock_trend_line_plot.png"
report_path = base / "stock_trend_report.md"

# Load
raw = pd.read_csv(input_path)

# Standardize column names
raw.columns = [c.strip() for c in raw.columns]

# Parse dates
raw['Date'] = pd.to_datetime(raw['Date'], errors='coerce')

# Coerce numeric columns
num_cols = ['Open','High','Low','Close','Volume']
for c in num_cols:
    raw[c] = pd.to_numeric(raw[c], errors='coerce')

# Drop rows with missing essentials
df = raw.dropna(subset=['Date'] + num_cols).copy()

# Remove duplicates by Date, keep last occurrence
if df['Date'].duplicated().any():
    df = df.sort_values('Date').drop_duplicates(subset=['Date'], keep='last')

# Basic validity checks
valid = (df['High'] >= df['Low'])
valid &= (df['Open'].between(df['Low'], df['High']))
valid &= (df['Close'].between(df['Low'], df['High']))
valid &= (df['Volume'] >= 0)

df = df[valid].copy()

# Sort by date
df = df.sort_values('Date')

# Save cleaned dataset
output_clean.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_clean, index=False)

# Compute trends
# Daily return based on Close
returns = df['Close'].pct_change()

summary = {}
summary['start_date'] = df['Date'].iloc[0].date()
summary['end_date'] = df['Date'].iloc[-1].date()
summary['start_close'] = df['Close'].iloc[0]
summary['end_close'] = df['Close'].iloc[-1]
summary['abs_change'] = summary['end_close'] - summary['start_close']
summary['pct_change'] = (summary['abs_change'] / summary['start_close']) * 100 if summary['start_close'] != 0 else np.nan
summary['avg_volume'] = df['Volume'].mean()
summary['volatility'] = returns.std() * 100

# Best/worst day by daily return
best_idx = returns.idxmax()
worst_idx = returns.idxmin()

# Moving average
df['MA_7'] = df['Close'].rolling(7, min_periods=1).mean()

# Plot
plt.figure(figsize=(10,5))
plt.plot(df['Date'], df['Close'], label='Close', linewidth=1.5)
plt.plot(df['Date'], df['MA_7'], label='7-day MA', linewidth=2)
plt.title('Stock Close Price Trend')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(plot_path, dpi=150)
plt.close()

# Report
with open(report_path, 'w', encoding='utf-8') as f:
    f.write('# Stock Trend Analysis (Time Series Line Plot)\n\n')
    f.write('## Dataset Summary\n')
    f.write(f"- Rows after cleaning: {len(df)}\n")
    f.write(f"- Date range: {summary['start_date']} to {summary['end_date']}\n")
    f.write('\n## Trend Highlights\n')
    f.write(f"- Close price change: {summary['abs_change']:.2f} ({summary['pct_change']:.2f}%)\n")
    f.write(f"- Average daily volume: {summary['avg_volume']:.0f}\n")
    f.write(f"- Volatility (std of daily returns): {summary['volatility']:.2f}%\n")
    if pd.notna(best_idx):
        f.write(f"- Best day: {df.loc[best_idx, 'Date'].date()} (Return {returns.loc[best_idx]*100:.2f}%)\n")
    if pd.notna(worst_idx):
        f.write(f"- Worst day: {df.loc[worst_idx, 'Date'].date()} (Return {returns.loc[worst_idx]*100:.2f}%)\n")
    f.write('\n## Outputs\n')
    f.write(f"- Cleaned dataset: {output_clean.name}\n")
    f.write(f"- Line plot: {plot_path.name}\n")

print('Cleaned rows:', len(df))
print('Saved:', output_clean)
print('Plot:', plot_path)
print('Report:', report_path)
