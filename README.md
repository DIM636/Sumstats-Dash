# Custom Simulation Results Analyzer

A web-based dashboard for analyzing and comparing simulation result files (e.g., `.out` files) with flexible stat selection, caching, and interactive visualization. Built with Python Dash and Plotly.

---

## Features

- 📂 **Directory-based batch analysis** of simulation result files
- 📝 **User-editable stat list** via `stat/stats.txt` (no code change needed)
- ⚡ **File-level caching** for fast repeated analysis
- 📊 **Interactive tables and graphs** (absolute values, performance change, drilldown)
- 🎨 **Pastel heatmap coloring** for intuitive comparison
- ⏱️ **Progress bar and analysis info** (immediate feedback)
- 🌐 **English UI** with clear messages and footer info
- 💾 **Save dashboard as HTML** (custom JS button)
- 📥 **Export tables as CSV**
- 🐧 **WSL/Windows/Linux compatible**

---

## Quick Start

### 1. Clone & Install

```bash
git clone <your-repo-url>
cd <repo-root>
pip install -r requirements.txt
```

### 2. Edit Stat List (Optional)

Edit `stat/stats.txt` to specify which stats to analyze (one per line):

```
ipc
power
total_power
```

### 3. Run the App

#### Development (with hot reload):
```bash
cd stat
python3 stat_dash.py
```

#### Production (recommended):
```bash
cd stat
bash run.sh
# or manually:
# gunicorn --bind 0.0.0.0:8000 --workers 2 stat_dash:server
```

Then open [http://localhost:8000](http://localhost:8000) in your browser.

---

## File Structure

```
stat/
  stat_dash.py        # Main Dash app
  stats.txt           # List of stats to analyze (editable)
  run.sh              # Gunicorn launch script
assets/
  custom_save.js      # JS for 'Save as HTML' button
requirements.txt      # Python dependencies
README.md             # This file
```

---

## Usage Tips

- **Select baseline and target directories** in the UI, or add custom paths manually.
- **Click 'Start Analysis'** to begin. Progress and info will show immediately.
- **Absolute Value Analysis**: See per-directory stat tables and graphs.
- **Performance Change Analysis**: Compare stats between baseline and targets, with heatmaps and drilldown.
- **Edit `stat/stats.txt`** to change which stats are analyzed (no code change needed).
- **Export**: Use the CSV export button in tables, or the 'Save as HTML' button for a full-page snapshot.

---

## Requirements

- Python 3.7+
- See `requirements.txt` for package versions

---

## Deployment & Environment

- **WSL/Windows/Linux** all supported
- For production, use Gunicorn (`run.sh`)
- If port 8000 is in use, change it in `run.sh`

---

## Author & License

- Created by dong63.ma
- MIT License (add your license if needed)

---

## Contact & Issues

- For questions or bug reports, please open a GitHub issue.
