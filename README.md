# Monaco Simulation Analyzer

A powerful web-based dashboard for analyzing and comparing Monaco simulation results with advanced statistical analysis capabilities.

## 🚀 Features

- **Multi-Directory Analysis**: Compare baseline vs target simulation results
- **Statistical Testing**: Significance testing with multiple comparison corrections
- **Effect Size Analysis**: Cohen's d calculation for practical significance
- **Interactive Drill-Down**: From summary to run-level data exploration
- **Natural Sorting**: Intelligent run name sorting (run1, run2, ..., run10)
- **Export Capabilities**: CSV export and HTML snapshot saving
- **Customizable Stats**: Edit analysis metrics via UI or stats.txt file
- **Customizable Color Directions**: Set color direction for each stat (🔴 Red for + or 🟢 Green for +)
- **Responsive Design**: Works on all screen sizes

## 📊 Analysis Tabs

### 1. Absolute Values Tab
- Raw performance metrics for each directory
- Interactive drill-down to run-level data
- Bar charts for visual comparison

### 2. Performance Change Tab
- Percentage changes with statistical significance (★)
- Effect size analysis (Cohen's d)
- Color-coded heatmaps for change magnitude
- Run-level drill-down for detailed analysis

### 3. Detailed Run Table Tab
- Side-by-side comparison of absolute values and changes
- Run-by-run analysis for selected group/subgroup
- Natural sorting of run names

## 🛠️ Quick Start

1. **Select Baseline Directory**: Choose reference simulation results
2. **Select Target Directories**: Choose directories to compare against baseline
3. **Configure Analysis**: 
   - Edit stat list (or modify stats.txt)
   - Enable statistical options (α level, correction method, effect size threshold)
   - Set color directions for each stat (🔴 Red for + or 🟢 Green for +)
4. **Start Analysis**: Click "Start Analysis" to process all .out files
5. **Explore Results**: Navigate through tabs to view different analysis perspectives

## 📁 Data Structure

```
study_output_directory/
├── baseline_dir/
│   ├── subgroup1/
│   │   ├── run1/
│   │   │   └── simulation.out
│   │   └── run2/
│   │       └── simulation.out
│   └── subgroup2/
│       └── ...
├── target_dir1/
│   └── ...
└── target_dir2/
    └── ...
```

## ⚙️ Configuration

### Environment Variables
- `STUDY_OUT_DIR`: Analysis root directory (default: current directory)

### Stats Configuration
Edit `stats.txt` or use the UI to customize analysis metrics:
```
ipc
power
L2_cache_miss_rate
total_power
```

### Statistical Options
- **Significance Level (α)**: 0.01 to 0.2 (default: 0.05)
- **Multiple Comparison Correction**: None, Bonferroni, Holm, FDR
- **Effect Size Threshold**: Cohen's d threshold for practical significance

### Color Direction Settings
- **Per-Stat Customization**: Set color direction for each statistic individually
- **🔴 Red for +**: Positive values shown in red (e.g., power, latency)
- **🟢 Green for +**: Positive values shown in green (e.g., ipc, throughput)
- **Default Settings**: Pre-configured for common metrics (ipc: green, power: red, etc.)

## 📈 Understanding Results

### Statistical Significance
- ★ indicates statistically significant changes (p-adj < α)
- Adjusted p-values account for multiple comparisons

### Effect Size (Cohen's d)
- ~0.2: Small effect
- ~0.5: Medium effect  
- ~0.8+: Large effect

### Color Coding
- **Customizable Direction**: Each stat can have its own color direction
- **🔴 Red for +**: Positive values in red (e.g., power consumption, latency)
- **🟢 Green for +**: Positive values in green (e.g., IPC, throughput)
- **Intensity**: Color intensity indicates magnitude of change
- **Default**: Pre-configured for intuitive interpretation

## 🔧 Installation

```bash
pip install dash dash-bootstrap-components pandas plotly scipy statsmodels
```

## 🚀 Usage

```bash
# Basic usage
python stat_dash.py

# With custom study directory
export STUDY_OUT_DIR=/path/to/results
python stat_dash.py

# Or specify via argument
python stat_dash.py --study_out_dir /path/to/results
```

Access the dashboard at `http://localhost:8000`

## 📋 Requirements

- Python 3.7+
- Dash
- Dash Bootstrap Components
- Pandas
- Plotly
- SciPy
- StatsModels

## 🎯 Tips

- Use sidebar toggle (◀/▶) to maximize viewing area
- Click table cells for run-level drill-down
- Export tables for external analysis
- Save HTML snapshots to preserve current view
- Adjust statistical options for more rigorous analysis

## 📄 License

Created by dong63.ma

## 🔄 Version History

- **v1.0.0**: Initial release with basic analysis capabilities
- **v1.1.0**: Added statistical testing and effect size analysis
- **v1.2.0**: Added Detailed Run Table and natural sorting
- **v1.3.0**: Enhanced drill-down functionality and UI improvements 