#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, Input, Output, State, MATCH, ALL, DiskcacheManager
import pandas as pd
import plotly.express as px
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import os
import time
import diskcache
import datetime
import numpy as np
import sys
import plotly
import base64
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests
import argparse
import hashlib
import mmap

# --- 컬러 팔레트 정의 ---
# [설명] 일관된 브랜드 컬러와 시각적 계층구조를 위한 컬러 팔레트
COLORS = {
    'primary': '#2c3e50',      # 진한 네이비 (메인 브랜드 컬러)
    'secondary': '#3498db',    # 밝은 블루 (보조 컬러)
    'accent': '#e74c3c',       # 빨강 (강조/경고)
    'success': '#27ae60',      # 초록 (성공/완료)
    'warning': '#f39c12',      # 주황 (경고)
    'info': '#17a2b8',         # 청록 (정보)
    'light': '#ecf0f1',        # 연한 회색 (배경)
    'dark': '#2c3e50',         # 진한 회색 (텍스트)
    'white': '#ffffff',        # 흰색
    'gray': '#95a5a6'          # 중간 회색
}

# Plotly 차트용 컬러 팔레트
PLOTLY_COLORS = ['#2c3e50', '#3498db', '#e74c3c', '#27ae60', '#f39c12', '#17a2b8', '#9b59b6', '#34495e']

# --- 기본 색상 방향 매핑 ---
# [설명] 각 통계 지표별로 양수일 때 어떤 색상을 사용할지 정의
# 'green_for_positive': 양수일 때 초록색 (기본값)
# 'red_for_positive': 양수일 때 빨간색
DEFAULT_COLOR_DIRECTIONS = {
    'ipc': 'green_for_positive',           # IPC는 높을수록 좋음 → 양수일 때 초록
    'power': 'red_for_positive',           # 전력은 낮을수록 좋음 → 양수일 때 빨강
    'L2_cache_miss_rate': 'red_for_positive',  # 캐시 미스율은 낮을수록 좋음 → 양수일 때 빨강
    'total_power': 'red_for_positive',     # 총 전력은 낮을수록 좋음 → 양수일 때 빨강
    'energy': 'red_for_positive',          # 에너지는 낮을수록 좋음 → 양수일 때 빨강
    'throughput': 'green_for_positive',    # 처리량은 높을수록 좋음 → 양수일 때 초록
    'latency': 'red_for_positive',         # 지연시간은 낮을수록 좋음 → 양수일 때 빨강
    'bandwidth': 'green_for_positive',     # 대역폭은 높을수록 좋음 → 양수일 때 초록
    'efficiency': 'green_for_positive',    # 효율성은 높을수록 좋음 → 양수일 때 초록
    'utilization': 'green_for_positive',   # 사용률은 높을수록 좋음 → 양수일 때 초록
}

# --- study_out_dir 지정 (실행 인자/환경변수 우선) ---
def get_study_out_dir():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study_out_dir', type=str, default=None)
    args, _ = parser.parse_known_args()
    study_out_dir = args.study_out_dir or os.environ.get('STUDY_OUT_DIR') or str(Path.cwd())
    return study_out_dir

STUDY_OUT_DIR = get_study_out_dir()

# --- 백그라운드 콜백 매니저 설정 ---
# [설명] 분석 작업의 캐시 및 비동기 처리를 위한 diskcache와 Dash의 백그라운드 콜백 매니저를 설정합니다.
# 캐시 경로를 /tmp 아래로 지정하여 권한 문제 방지
cache_directory = "/tmp/my_dash_analyzer_cache"
os.makedirs(cache_directory, exist_ok=True)
cache = diskcache.Cache(cache_directory)
background_callback_manager = DiskcacheManager(cache)

# --- 분석 로직 ---
# [설명] 결과 파일(.out)에서 stat 값을 추출하고, DataFrame으로 집계하는 주요 분석 함수들입니다.
def load_stats_to_find(stats_file='stats.txt'):
    stats_path = Path(stats_file)
    if not stats_path.is_file():
        # 기본값 제공 (파일 없을 때)
        return ['ipc', 'power', 'L2_cache_miss_rate', 'total_power']
    with open(stats_path, 'r', encoding='utf-8') as f:
        stats = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
    return stats

STATS_TO_FIND = load_stats_to_find('stats.txt')
KEY_VALUE_PATTERN = re.compile(r"^\s*(\S+)\s*=\s*([-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?|[Nn][Aa][Nn])")

def natural_key(s):
    """자연스러운 정렬을 위한 키 함수 (run1, run2, ..., run10 순서)"""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', str(s))]

def find_subdirectories(base_path: Path, depth: int) -> list:
    # [설명] 기준 경로에서 지정한 depth까지 하위 디렉토리를 탐색해 목록을 반환합니다.
    dirs = set()
    if not base_path.is_dir(): return []
    with os.scandir(base_path) as it:
        for entry in it:
            if entry.is_dir():
                dirs.add(str(Path(entry.path).resolve()))
                if depth > 1:
                    try:
                        with os.scandir(entry.path) as sub_it:
                            for sub_entry in sub_it:
                                if sub_entry.is_dir():
                                    dirs.add(str(Path(sub_entry.path).resolve()))
                    except PermissionError:
                        continue
    return sorted(list(dirs))

# --- Dash 앱 구성 ---
# [설명] Dash 앱 객체, 서버, 타이틀, 외부 스타일, 백그라운드 콜백 매니저 등을 설정합니다.
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], background_callback_manager=background_callback_manager)
server = app.server
app.title = "Monaco Simulation Analyzer"
discovered_dirs = find_subdirectories(Path(STUDY_OUT_DIR), depth=5)
dir_options = [{'label': os.path.basename(p), 'value': p} for p in discovered_dirs]

# 1. 사이드바 controls에 stat 목록 편집 UI 추가
controls = dbc.Card([
    dbc.CardHeader([
        html.H4("📊 Analysis Configuration", className="card-title mb-0", 
                style={"color": COLORS['primary'], "fontWeight": "600"})
    ], style={"backgroundColor": COLORS['light'], "borderBottom": f"3px solid {COLORS['secondary']}"}),
    dbc.CardBody([
        # study_out_dir 입력 및 Search 버튼
        html.Div([
            html.H5("Study Output Directory", style={"color": COLORS['primary'], "fontWeight": "600", "marginBottom": "10px"}),
            dcc.Input(id='study-out-dir-input', type='text', value=STUDY_OUT_DIR, style={"width": "80%", "marginRight": "8px"}),
            dbc.Button("Search", id="study-out-dir-search-btn", color="secondary", size="sm"),
            html.P(id='study-out-dir-status', style={"fontSize": "0.95em", "color": COLORS['gray'], "marginTop": "5px"}),
        ], style={"marginBottom": "18px"}),
        # stat 목록 편집
        html.Div([
            html.H5("Edit Stat List", style={"color": COLORS['primary'], "fontWeight": "600", "marginBottom": "10px"}),
            dcc.Textarea(id='stat-list-input', value='\n'.join(STATS_TO_FIND), style={"width": "100%", "height": "80px", "borderColor": COLORS['secondary']}),
            dbc.Button("Apply", id="stat-list-apply-btn", color="secondary", size="sm", className="mt-2 me-2"),
            dbc.Button("Reset to Default", id="stat-list-reset-btn", color="light", size="sm", className="mt-2"),
            html.P(id='stat-list-status', style={"fontSize": "0.95em", "color": COLORS['gray'], "marginTop": "5px"}),
        ], style={"marginBottom": "18px"}),
        # Section 1: Baseline Directory
        html.Div([
            html.H5("1. Select Baseline Directory", 
                    style={"color": COLORS['primary'], "fontWeight": "600", "marginBottom": "15px"}),
            html.Div([
                html.P("This tool analyzes the following stats:", 
                       style={"fontWeight": "600", "color": COLORS['dark'], "marginBottom": "10px"}),
                html.Ul(id='baseline-stats-list', children=[html.Li(stat, style={"color": COLORS['dark'], "marginBottom": "5px"}) for stat in STATS_TO_FIND], 
                       style={"backgroundColor": COLORS['light'], "padding": "10px", "borderRadius": "5px"}),
                html.P("Click 'Start Analysis' to automatically aggregate and compare the above stats for the selected directories.", 
                       style={"fontSize": "0.95em", "color": COLORS['gray'], "fontStyle": "italic"})
            ], className="mb-4"),
            dcc.Dropdown(id='baseline-dropdown', options=dir_options, 
                        placeholder="Select the baseline directory...",
                        style={"borderColor": COLORS['secondary']})
        ], style={"marginBottom": "25px"}),
        
        # Section 2: Target Directories
        html.Div([
            html.H5("2. Select Target Directories", 
                    style={"color": COLORS['primary'], "fontWeight": "600", "marginBottom": "15px"}),
            dcc.Checklist(id='target-checklist', options=dir_options, 
                         labelClassName="me-3", inputClassName="me-1",
                         style={"color": COLORS['dark']})
        ], style={"marginBottom": "25px"}),
        
        # Section 3: Manual Paths
        html.Div([
            html.H5("3. (Optional) Add Directories Manually", 
                    style={"color": COLORS['primary'], "fontWeight": "600", "marginBottom": "15px"}),
            dbc.Textarea(id="manual-path-input", 
                        placeholder="Enter directory paths not listed above, one per line...", 
                        style={'height': '80px', "borderColor": COLORS['secondary']})
        ], style={"marginBottom": "25px"}),

        # Section 4: Statistical Test Options
        html.Div([
            html.H5("4. Statistical Test Options", 
                    style={"color": COLORS['primary'], "fontWeight": "600", "marginBottom": "15px"}),
            # 추가: 통계 옵션 활성화 체크박스
            dbc.Checkbox(
                id='enable-statistical-options',
                value=False,
                label="Enable statistical test & effect size analysis",
                style={"marginBottom": "10px", "fontWeight": "600", "color": COLORS['secondary']}
            ),
            html.Div([
                html.Label("Significance Level (α):", style={"fontWeight": "600", "marginRight": "10px"}),
                dcc.Slider(
                    id='alpha-slider',
                    min=0.01, max=0.2, step=0.01, value=0.05,
                    marks={0.01: '0.01', 0.05: '0.05', 0.1: '0.1', 0.2: '0.2'},
                    tooltip={"placement": "bottom", "always_visible": False},
                    included=False,
                    updatemode='drag',
                    className="mb-2",
                    disabled=True  # 기본 비활성화
                ),
            ], style={"marginBottom": "18px"}),
            html.Div([
                html.Label("Multiple Comparison Correction:", style={"fontWeight": "600", "marginRight": "10px"}),
                dcc.Dropdown(
                    id='correction-method-dropdown',
                    options=[
                        {"label": "None", "value": "none"},
                        {"label": "Bonferroni", "value": "bonferroni"},
                        {"label": "Holm", "value": "holm"},
                        {"label": "Benjamini-Hochberg (FDR)", "value": "fdr_bh"}
                    ],
                    value="none",
                    clearable=False,
                    style={"width": "70%"},
                    disabled=True  # 기본 비활성화
                )
            ], style={"marginBottom": "18px"}),
            html.Div([
                html.Label("Effect Size Threshold (Cohen's d):", style={"fontWeight": "600", "marginRight": "10px"}),
                dcc.Slider(
                    id='effect-size-threshold-slider',
                    min=0, max=2, step=0.05, value=0,
                    marks={0: '0', 0.2: '0.2', 0.5: '0.5', 0.8: '0.8', 1.0: '1.0', 1.5: '1.5', 2.0: '2.0'},
                    tooltip={"placement": "bottom", "always_visible": False},
                    included=False,
                    updatemode='drag',
                    className="mb-2",
                    disabled=True  # 기본 비활성화
                ),
            ])
        ], style={"marginBottom": "25px"}),

        # Section 5: Color Direction Settings
        html.Div([
            html.H5("5. 🎨 Color Direction Settings", 
                    style={"color": COLORS['primary'], "fontWeight": "600", "marginBottom": "15px"}),
            html.P("Set color direction for each stat in change tables:", 
                   style={"fontSize": "0.9em", "color": COLORS['gray'], "marginBottom": "10px"}),
            html.Div(id='color-direction-controls', style={"marginBottom": "10px"}),
            dbc.Button("Apply Colors", id="apply-colors-btn", color="secondary", size="sm", className="mt-2"),
            html.P(id='color-direction-status', 
                   style={"fontSize": "0.9em", "color": COLORS['gray'], "marginTop": "5px"}),
        ], style={"marginBottom": "25px"}),

        # Analysis Button
        dbc.Button("🚀 Start Analysis", id="run-button", 
                  color="primary", className="my-3 w-100",
                  style={"backgroundColor": COLORS['secondary'], "borderColor": COLORS['secondary'], 
                         "fontWeight": "600", "fontSize": "1.1em", "padding": "12px"}),
        dbc.Tooltip(
            "Click to start analyzing the selected directories. This will process all .out files and generate comparison reports.",
            target="run-button",
            placement="top"
        ),
        
        # Progress and Info
        html.Div(id="progress-wrapper", 
                children=[dbc.Progress(id="progress-bar", value=100, striped=True, animated=True)], 
                style={'display': 'none'}),
        html.Div(id="analysis-info-start", className="mt-3"),
        html.Div(id="analysis-info-complete", className="mt-3"),
        # Footer
        html.Footer([
            html.Hr(style={"borderColor": COLORS['light'], "margin": "30px 0"}),
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Span("Monaco Simulation Analyzer — Created by dong63.ma", 
                                  style={"color": COLORS['primary'], "fontWeight": "600", "marginRight": "15px"}),
                        html.Span("| Version: v1.0.0", 
                                  style={"color": COLORS['gray'], "marginRight": "15px"}),
                        html.Span("| Last updated: 2025-07-21", 
                                  style={"color": COLORS['gray'], "marginRight": "15px"}),
                        html.Span(f"| Python {sys.version_info.major}.{sys.version_info.minor} | Dash {dash.__version__} | Plotly {plotly.__version__}", 
                                  style={"color": COLORS['gray']})
                    ], style={"textAlign": "center", "fontSize": "0.95em"})
                ], width=12)
            ])
        ])
    ])
], style={"border": f"1px solid {COLORS['light']}", "boxShadow": "0 2px 10px rgba(0,0,0,0.1)"})

# --- percentage-change-tab 상단 옵션 UI 추가 ---
stat_options = [{'label': stat, 'value': stat} for stat in STATS_TO_FIND]

app.layout = dbc.Container([
    dbc.Row([
        # Sidebar (좌측) - 접힘 가능
        dbc.Col([
            # 접힘 버튼 (항상 보임)
            dbc.Row([
                dbc.Col([
                    dbc.Button("◀", id="sidebar-toggle", color="light", size="sm", 
                              style={"backgroundColor": COLORS['white'], "borderColor": COLORS['gray'], 
                                     "color": COLORS['dark'], "fontWeight": "bold", "float": "right"})
                ], width=12, className="text-end mb-2")
            ]),
            # 사이드바 헤더 (접힘 가능)
            dbc.Collapse([
                html.H2("Options", className="display-6 mb-3", style={"color": COLORS['primary'], "fontWeight": "700"}),
                html.Hr(),
            ], id="sidebar-header", is_open=True),
            # 사이드바 내용 (접힘 가능)
            dbc.Collapse([
                controls,
                # 도움말 버튼 추가
                html.Hr(className="my-4"),
                dbc.Button("❓ Help Guide", id="help-button", color="info", size="lg", className="w-100",
                          style={"backgroundColor": COLORS['info'], "borderColor": COLORS['info'], "fontWeight": "600"}),
            ], id="sidebar-content", is_open=True),
        ], id="sidebar-col", width=3, style={
            "backgroundColor": COLORS['light'],
            "minHeight": "100vh",
            "padding": "30px 15px",
            "boxShadow": "2px 0 8px rgba(0,0,0,0.04)",
            "transition": "all 0.3s ease"
        }),
        # Main content (우측) - 반응형
        dbc.Col([
            # Header
            html.H1("📊 Monaco Simulation Results Analyzer", 
                    className="text-center mb-3",
                    style={"color": COLORS['primary'], "fontWeight": "700", "fontSize": "2.5em"}),
            html.P("Advanced simulation data analysis and comparison dashboard", 
                   className="text-center mb-4",
                   style={"color": COLORS['gray'], "fontSize": "1.1em", "fontStyle": "italic"}),
            # Save HTML Button (오른쪽 상단, 작게)
            html.Div([
                dbc.Button(
                    "💾 Save as HTML", id="save-html-btn",
                    color="secondary", size="sm", outline=True,
                    className="float-end mt-2 me-2",
                    style={"fontWeight": "600", "boxShadow": "0 1px 4px rgba(0,0,0,0.07)"}
                ),
                dbc.Tooltip(
                    "Save the current dashboard view as an HTML file (including graphs, filters, etc.)",
                    target="save-html-btn",
                    placement="left"
                ),
                html.Div(id="save-status", className="float-end me-2", style={"fontSize": "0.95em", "marginTop": "2.5rem"})
            ], style={"minHeight": "40px", "position": "relative"}),
            # Main Tabs
            dbc.Tabs(id="tabs-container", children=[
                dbc.Tab(html.Div(id="absolute-values-tab"), label="📈 Absolute Values", 
                        tab_id="absolute", label_style={"color": COLORS['primary'], "fontWeight": "600"}),
                dbc.Tab(html.Div(id="percentage-change-tab"), label="📊 Performance Change", 
                        tab_id="change", label_style={"color": COLORS['primary'], "fontWeight": "600"}),
                dbc.Tab(
                    html.Div([
                        html.H3("Detailed Run Table", className="mt-3"),
                        dcc.Dropdown(id='detailed-group-subgroup-dropdown', options=[], placeholder="Select test group/subgroup...", style={"marginBottom": "20px"}),
                        html.Div(id='detailed-run-table-container')
                    ]),
                    label="🧾 Detailed Run Table",
                    tab_id="detailed-run-tab",
                    label_style={"color": COLORS['primary'], "fontWeight": "600"}
                ),
            ], style={"marginBottom": "30px"}),
            # Stores (hidden)
            dcc.Store(id='job-start-time-store'),
            dcc.Store(id='summary-data-store'),
            dcc.Store(id='change-data-store'),
            dcc.Store(id='dir-names-store'),
            dcc.Store(id='run-change-data-store'),
            dcc.Store(id='analysis-meta-store'),
            dcc.Interval(id='progress-interval', interval=1000, disabled=True),
            dcc.Store(id='stats-store', data=STATS_TO_FIND),
            dcc.Store(id='stat-matched-keys-store'), # 추가: stat-matched-keys-store
            dcc.Store(id='analysis-complete-store'), # 분석 완료 전용 Store
            dcc.Store(id='study-out-dir-store', data=STUDY_OUT_DIR), # 추가: study_out_dir Store
            dcc.Store(id='color-direction-store', data=DEFAULT_COLOR_DIRECTIONS), # 추가: 색상 방향 설정 Store
            # Footer
            html.Footer([
                html.Hr(style={"borderColor": COLORS['light'], "margin": "30px 0"}),
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            html.Span("Monaco Simulation Analyzer — Created by dong63.ma", 
                                      style={"color": COLORS['primary'], "fontWeight": "600", "marginRight": "15px"}),
                            html.Span("| Version: v1.0.0", 
                                      style={"color": COLORS['gray'], "marginRight": "15px"}),
                            html.Span("| Last updated: 2025-01-20", 
                                      style={"color": COLORS['gray'], "marginRight": "15px"}),
                            html.Span(f"| Python {sys.version_info.major}.{sys.version_info.minor} | Dash {dash.__version__} | Plotly {plotly.__version__}", 
                                      style={"color": COLORS['gray']})
                        ], style={"textAlign": "center", "fontSize": "0.95em"})
                    ], width=12)
                ])
            ])
        ], width=9, style={"padding": "30px 30px", "backgroundColor": COLORS['white']})
    ], className="g-0", style={"minHeight": "100vh"}),
    
    # 도움말 모달
    dbc.Modal([
        dbc.ModalHeader([
            dbc.ModalTitle("📖 Monaco Simulation Analyzer - User Guide", 
                           style={"color": COLORS['primary'], "fontWeight": "600"})
        ], style={"backgroundColor": COLORS['light']}),
        dbc.ModalBody([
            html.H5("🚀 Quick Start", style={"color": COLORS['primary'], "fontWeight": "600", "marginBottom": "15px"}),
            html.Ol([
                html.Li([
                    html.Strong("Select Baseline Directory: "),
                    "Choose the reference directory containing your baseline simulation results"
                ], style={"marginBottom": "8px"}),
                html.Li([
                    html.Strong("Select Target Directories: "),
                    "Choose one or more directories to compare against the baseline"
                ], style={"marginBottom": "8px"}),
                html.Li([
                    html.Strong("Optional: Add Manual Paths: "),
                    "Enter additional directory paths not listed in the dropdown"
                ], style={"marginBottom": "8px"}),
                html.Li([
                    html.Strong("Edit Stat List: "),
                    "Modify the list of statistics to analyze (or edit stats.txt file)"
                ], style={"marginBottom": "8px"}),
                html.Li([
                    html.Strong("Configure Statistical Options: "),
                    "Enable significance testing, set α level, correction method, and effect size threshold"
                ], style={"marginBottom": "8px"}),
                html.Li([
                    html.Strong("Set Color Directions: "),
                    "Choose color direction for each stat (🔴 Red for + or 🟢 Green for +) in change tables"
                ], style={"marginBottom": "8px"}),
                html.Li([
                    html.Strong("Start Analysis: "),
                    "Click 'Start Analysis' to process all .out files and generate reports"
                ], style={"marginBottom": "8px"}),
                html.Li([
                    html.Strong("Explore Results: "),
                    "View absolute values, performance changes, and detailed run-level data in the tabs"
                ], style={"marginBottom": "8px"})
            ], style={"marginBottom": "20px"}),

            html.H5("📊 Understanding Results", style={"color": COLORS['primary'], "fontWeight": "600", "marginBottom": "15px"}),
            html.Ul([
                html.Li([
                    html.Strong("Absolute Values Tab: "),
                    "Raw performance metrics for each directory with drill-down to run-level data"
                ], style={"marginBottom": "8px"}),
                html.Li([
                    html.Strong("Performance Change Tab: "),
                    "Percentage changes with statistical significance (★) and effect size (Cohen's d)"
                ], style={"marginBottom": "8px"}),
                html.Li([
                    html.Strong("Detailed Run Table Tab: "),
                    "Run-by-run comparison showing absolute values and changes side by side"
                ], style={"marginBottom": "8px"}),
                html.Li([
                    html.Strong("Statistical Significance: "),
                    "★ indicates statistically significant changes (p-adj < α)"
                ], style={"marginBottom": "8px"}),
                html.Li([
                    html.Strong("Effect Size: "),
                    "Cohen's d: ~0.2 (small), ~0.5 (medium), ~0.8+ (large)"
                ], style={"marginBottom": "8px"}),
                html.Li([
                    html.Strong("Color Coding: "),
                    "Green/red pastel colors indicate improvement/regression magnitude. Customizable direction per stat"
                ], style={"marginBottom": "8px"})
            ], style={"marginBottom": "20px"}),

            html.H5("⚙️ Key Features", style={"color": COLORS['primary'], "fontWeight": "600", "marginBottom": "15px"}),
            html.Ul([
                html.Li("Automatic .out file discovery and parsing"),
                html.Li("Multi-processor analysis for fast processing"),
                html.Li("Statistical significance testing with multiple comparison corrections"),
                html.Li("Effect size analysis (Cohen's d)"),
                html.Li("Interactive drill-down from summary to run-level data"),
                html.Li("CSV export for all tables"),
                html.Li("HTML snapshot saving"),
                html.Li("Customizable stat list"),
                html.Li("Natural sorting of run names (run1, run2, ..., run10)"),
                html.Li("Responsive design for all screen sizes")
            ], style={"marginBottom": "20px"}),

            html.H5("💡 Tips", style={"color": COLORS['primary'], "fontWeight": "600", "marginBottom": "15px"}),
            html.Ul([
                html.Li("Use the sidebar toggle (◀/▶) to maximize viewing area"),
                html.Li("Click on table cells to drill down to run-level details"),
                html.Li("Adjust statistical options for more rigorous analysis"),
                html.Li("Export tables for further analysis in external tools"),
                html.Li("Save HTML snapshots to preserve current view with all filters")
            ])
        ]),
        dbc.ModalFooter([
            dbc.Button("Close", id="close-help", className="ms-auto", 
                      style={"backgroundColor": COLORS['secondary'], "borderColor": COLORS['secondary']})
        ])
    ], id="help-modal", is_open=False, size="lg")
], fluid=True)

# --- 콜백 함수들 ---
# [설명] 사용자 인터랙션(분석 시작, 진행/완료 표시, 데이터 분석, 시각화 등)에 따라 동적으로 UI를 업데이트하는 Dash 콜백 함수들입니다.

# 통계 옵션 활성화 체크박스에 따라 슬라이더/드롭다운 활성화
@app.callback(
    [Output('alpha-slider', 'disabled'),
     Output('correction-method-dropdown', 'disabled'),
     Output('effect-size-threshold-slider', 'disabled')],
    Input('enable-statistical-options', 'value')
)
def toggle_statistical_options(enabled):
    disabled = not enabled
    return [disabled, disabled, disabled]

# 콜백 1: '분석 시작' 버튼 클릭 시, 스토어 초기화, 진행률 바 활성화, 버튼 비활성화
@app.callback(
    [Output('job-start-time-store', 'data'),
     Output('progress-interval', 'disabled'),
     Output('progress-wrapper', 'style'),
     Output('run-button', 'disabled'),
     Output('summary-data-store', 'data', allow_duplicate=True),
     Output('change-data-store', 'data', allow_duplicate=True),
     Output('dir-names-store', 'data', allow_duplicate=True),
     Output('analysis-meta-store', 'data')],
    Input('run-button', 'n_clicks'),
    State('baseline-dropdown', 'value'),
    State('target-checklist', 'value'),
    State('manual-path-input', 'value'),
    State('study-out-dir-store', 'data'), # 추가
    prevent_initial_call=True,
)
def start_analysis_job(n_clicks, baseline_dir, target_dirs_checked, manual_paths, study_out_dir):
    import datetime, time, os
    start_time = time.time()
    def resolve_path(p):
        if not p:
            return None
        if os.path.isabs(p):
            return p
        return str(Path(study_out_dir) / p)
    dirs = [resolve_path(baseline_dir)] if baseline_dir else []
    if target_dirs_checked:
        dirs += [resolve_path(p) for p in target_dirs_checked]
    if manual_paths:
        dirs += [resolve_path(p.strip()) for p in manual_paths.split('\n') if p.strip()]
    dirs = [d for d in dict.fromkeys(dirs) if d]
    start_str = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    meta = {'dirs': dirs, 'start_str': start_str}
    # summary-data-store.data를 항상 None으로 리셋
    return start_time, False, {'display': 'block'}, True, None, None, None, meta

# 콜백 2: [백그라운드] 실제 분석을 수행하고 결과를 dcc.Store에 저장
def cohens_d(x, y):
    # 두 집단의 Cohen's d
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    mean_x, mean_y = np.mean(x), np.mean(y)
    s1, s2 = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_std = np.sqrt(((nx - 1) * s1 + (ny - 1) * s2) / (nx + ny - 2))
    if pooled_std == 0:
        return 0.0
    return (mean_x - mean_y) / pooled_std

# 2. stat 목록 편집 콜백
@app.callback(
    [Output('stat-list-status', 'children'),
     Output('stat-list-input', 'value'),
     Output('stats-store', 'data')],
    [Input('stat-list-apply-btn', 'n_clicks'),
     Input('stat-list-reset-btn', 'n_clicks')],
    State('stat-list-input', 'value'),
    prevent_initial_call=True
)
def update_stats_list(apply_n, reset_n, stat_text):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise dash.exceptions.PreventUpdate
    btn_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if btn_id == 'stat-list-reset-btn':
        default_stats = load_stats_to_find('stats.txt')
        return "✅ Stat list reset to default.", '\n'.join(default_stats), default_stats
    # Apply 버튼
    stats = [s.strip() for s in stat_text.splitlines() if s.strip()]
    if not stats:
        return "❌ Please enter at least one stat.", stat_text, dash.no_update
    return f"✅ Stat list updated: {', '.join(stats)}", '\n'.join(stats), stats

# stats-store 변경 시 Baseline Directory stat 리스트 동적 업데이트 콜백
@app.callback(
    Output('baseline-stats-list', 'children'),
    [Input('stats-store', 'data'),
     Input('stat-matched-keys-store', 'data')]
)
def update_baseline_stats_list(stats, matched_keys):
    if not stats:
        return []
    items = []
    for stat in stats:
        keys = matched_keys.get(stat, []) if matched_keys else []
        tooltip = f"Matched keys: {', '.join(keys)}" if keys else "No matched keys yet. Will be shown after analysis."
        items.append(
            html.Li([
                html.Span(stat, id={'type': 'stat-tooltip', 'stat': stat}, style={"color": COLORS['dark'], "marginBottom": "5px"}),
                dbc.Tooltip(tooltip, target={'type': 'stat-tooltip', 'stat': stat}, placement="right")
            ], style={"marginBottom": "5px"})
        )
    return items

# 3. 분석 콜백에서 stats-store의 값을 사용
@app.callback(
    [Output('summary-data-store', 'data'),
     Output('change-data-store', 'data'),
     Output('dir-names-store', 'data'),
     Output('run-change-data-store', 'data'),
     Output('stat-matched-keys-store', 'data'),
     Output('analysis-complete-store', 'data')],
    Input('run-button', 'n_clicks'),
    [State("baseline-dropdown", "value"),
     State("target-checklist", "value"),
     State("manual-path-input", "value"),
     State("alpha-slider", "value"),
     State("correction-method-dropdown", "value"),
     State("effect-size-threshold-slider", "value"),
     State('stats-store', 'data'),
     State('enable-statistical-options', 'value'),
     State('study-out-dir-store', 'data')], # 추가
    background=True,
    manager=background_callback_manager,
    prevent_initial_call=True,
)
def run_analysis_callback(set_progress, baseline_dir, target_dirs_checked, manual_paths, alpha, correction_method, effect_size_threshold, stats_to_find, enable_statistical_options, study_out_dir):
    import os
    def resolve_path(p):
        if not p:
            return None
        if os.path.isabs(p):
            return p
        return str(Path(study_out_dir) / p)
    if not baseline_dir: return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    targets = set(target_dirs_checked) if target_dirs_checked else set()
    if manual_paths:
        for path in manual_paths.split('\n'):
            if path.strip(): targets.add(path.strip())
    targets.discard(baseline_dir)
    if not targets: return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update
    if not stats_to_find:
        stats_to_find = STATS_TO_FIND
    root_dirs = [resolve_path(baseline_dir)] + sorted([resolve_path(p) for p in targets])
    dir_names = [os.path.basename(p) for p in root_dirs]
    summaries = []
    all_matched_keys = {stat: set() for stat in stats_to_find}
    for dir_path in root_dirs:
        df, matched_keys = analyze_directory(Path(dir_path), stats_to_find)
        summaries.append(df)
        for stat, keys in matched_keys.items():
            all_matched_keys[stat].update(keys)
    summaries_json = [s.to_json(orient='split') if s is not None else None for s in summaries]
    change_dfs_json = []
    run_change_dfs_json = []
    if summaries[0] is not None:
        for i in range(1, len(summaries)):
            if summaries[i] is not None:
                base = summaries[0].groupby(['Subgroup', 'Stat'])['Value'].mean().reset_index()
                targ = summaries[i].groupby(['Subgroup', 'Stat'])['Value'].mean().reset_index()
                merged = pd.merge(targ, base, on=['Subgroup', 'Stat'], suffixes=('_target', '_baseline'), how='outer')
                merged['Change'] = ((merged['Value_target'] - merged['Value_baseline']) / merged['Value_baseline']) * 100
                if enable_statistical_options:
                    # --- p-value, effect size 계산 ---
                    pvals = []
                    ds = []
                    for _, row in merged.iterrows():
                        sub, stat = row['Subgroup'], row['Stat']
                        base_vals = summaries[0][(summaries[0]['Subgroup'] == sub) & (summaries[0]['Stat'] == stat)]['Value'].values
                        targ_vals = summaries[i][(summaries[i]['Subgroup'] == sub) & (summaries[i]['Stat'] == stat)]['Value'].values
                        if len(base_vals) > 1 and len(targ_vals) > 1:
                            try:
                                stat_res = ttest_ind(base_vals, targ_vals, equal_var=False, nan_policy='omit')
                                pval = stat_res.pvalue
                                d = cohens_d(targ_vals, base_vals)
                            except Exception:
                                pval = np.nan
                                d = np.nan
                        else:
                            pval = np.nan
                            d = np.nan
                        pvals.append(pval)
                        ds.append(d)
                    merged['p-value'] = pvals
                    merged['effect_size'] = ds
                    # --- 다중비교 보정 ---
                    if correction_method and correction_method != 'none':
                        mask = ~pd.isna(merged['p-value'])
                        pvals_arr = merged.loc[mask, 'p-value'].values
                        if len(pvals_arr) > 0:
                            reject, pvals_corr, _, _ = multipletests(pvals_arr, alpha=alpha, method=correction_method)
                            merged.loc[mask, 'p-adj'] = pvals_corr
                            merged.loc[mask, 'signif'] = reject
                        else:
                            merged['p-adj'] = np.nan
                            merged['signif'] = False
                    else:
                        merged['p-adj'] = merged['p-value']
                        merged['signif'] = merged['p-value'] < alpha
                    merged['star'] = merged['signif'].apply(lambda x: '*' if x else '')
                # 피벗: 변화율, p-adj, 별표, effect_size
                pivot_change = merged.pivot(index='Subgroup', columns='Stat', values='Change').reset_index().round(2)
                if enable_statistical_options:
                    pivot_padj = merged.pivot(index='Subgroup', columns='Stat', values='p-adj').reset_index().round(4)
                    pivot_star = merged.pivot(index='Subgroup', columns='Stat', values='star').reset_index()
                    pivot_d = merged.pivot(index='Subgroup', columns='Stat', values='effect_size').reset_index().round(3)
                columns = ['Subgroup']
                for stat in pivot_change.columns:
                    if stat != 'Subgroup':
                        columns.append((stat, 'Change'))
                        if enable_statistical_options:
                            columns.extend([(stat, 'p-adj'), (stat, 'Signif'), (stat, 'd')])
                data = []
                for idx in range(len(pivot_change)):
                    row = {'Subgroup': pivot_change.loc[idx, 'Subgroup']}
                    for stat in pivot_change.columns:
                        if stat == 'Subgroup':
                            continue
                        val = pivot_change.loc[idx, stat]
                        row[(stat, 'Change')] = val if pd.notna(val) else 'N/A'
                        if enable_statistical_options:
                            padj_val = pivot_padj.loc[idx, stat] if stat in pivot_padj.columns else np.nan
                            star_val = pivot_star.loc[idx, stat] if stat in pivot_star.columns else ''
                            d_val = pivot_d.loc[idx, stat] if stat in pivot_d.columns else np.nan
                            row[(stat, 'p-adj')] = padj_val if pd.notna(padj_val) else 'N/A'
                            row[(stat, 'Signif')] = star_val if pd.notna(star_val) else 'N/A'
                            row[(stat, 'd')] = d_val if pd.notna(d_val) else 'N/A'
                    data.append(row)
                table_df = pd.DataFrame(data, columns=columns)
                # 평균(min) 행 추가
                stat_cols = [col for col in table_df.columns if col != 'Subgroup' and (col[1] == 'Change')]
                min_row = {'Subgroup': 'mean'}
                for stat_col in stat_cols:
                    min_row[stat_col] = table_df[stat_col].mean()
                table_df = pd.concat([table_df, pd.DataFrame([min_row])], ignore_index=True)
                change_dfs_json.append(table_df.to_json(orient='split'))
                # run별 변화율 계산 (기존과 동일)
                base_run = summaries[0][['Subgroup', 'Run', 'Stat', 'Value']].copy()
                targ_run = summaries[i][['Subgroup', 'Run', 'Stat', 'Value']].copy()
                run_merged = pd.merge(targ_run, base_run, on=['Subgroup', 'Run', 'Stat'], suffixes=('_target', '_baseline'))
                run_merged['Change'] = ((run_merged['Value_target'] - run_merged['Value_baseline']) / run_merged['Value_baseline']) * 100
                run_change_dfs_json.append(run_merged.to_json(orient='split'))
            else:
                change_dfs_json.append(None)
                run_change_dfs_json.append(None)
    # 마지막 Output: 분석 완료 신호 True
    return summaries_json, change_dfs_json, dir_names, run_change_dfs_json, {stat: list(keys) for stat, keys in all_matched_keys.items()}, True

# 콜백 3: [실시간 업데이트] Interval을 통해 수행 시간 업데이트 및 완료 처리
@app.callback(
    [Output('progress-bar', 'label'),
     Output('progress-interval', 'disabled', allow_duplicate=True),
     Output('progress-wrapper', 'style', allow_duplicate=True),
     Output('run-button', 'disabled', allow_duplicate=True)],
    [Input('progress-interval', 'n_intervals'),
     Input('summary-data-store', 'data')],
    State('job-start-time-store', 'data'),
    prevent_initial_call=True,
)
def update_progress_label(n_intervals, summary_data, start_time):
    if summary_data:
        elapsed_time = time.time() - start_time
        td = datetime.timedelta(seconds=int(elapsed_time))
        label = f"Analysis completed! (Total elapsed time: {td})"
        return label, True, {'display': 'none'}, False
    elapsed_time = time.time() - start_time
    td = datetime.timedelta(seconds=int(elapsed_time))
    label = f"Analysis in progress... (Elapsed time: {td})"
    return label, False, {'display': 'block'}, True

# 콜백 4: Store의 절대값 데이터로 '절대값 분석' 탭 내용을 생성
@app.callback(
    Output("absolute-values-tab", "children"),
    [Input("summary-data-store", "data"),
     State("dir-names-store", "data"),
     State('stats-store', 'data'),
     State('stat-matched-keys-store', 'data')]
)
def update_absolute_values_tab(summaries_json, dir_names, stats_to_find, matched_keys):
    if summaries_json is None:
        return "Please start analysis or wait for it to complete."
    output_components = []
    for i, summary_json in enumerate(summaries_json):
        if summary_json is None:
            continue
        dir_name = dir_names[i]
        summary_df = pd.read_json(summary_json, orient='split')
        output_components.append(html.H3(f"📁 {dir_name} Absolute Value Analysis", className="mt-4"))
        # 파일명 생성
        now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        # subgroup별 stat별 평균 테이블
        if {'Subgroup', 'Stat', 'Value'}.issubset(summary_df.columns):
            pivot_df = summary_df.groupby(['Subgroup', 'Stat'])['Value'].mean().reset_index()
            table_df = pivot_df.pivot(index='Subgroup', columns='Stat', values='Value')
            # stat-list 기준 컬럼 보장
            for stat in stats_to_find:
                if stat not in table_df.columns:
                    table_df[stat] = np.nan
            table_df = table_df[stats_to_find]
            table_df = table_df.reset_index().round(4)
            # 평균(min) 행 추가
            stat_cols = [col for col in table_df.columns if col != 'Subgroup']
            min_row = {'Subgroup': 'mean'}
            for stat in stat_cols:
                min_row[stat] = table_df[stat].mean()  # 평균, min으로 바꾸려면 .min()
            table_df = pd.concat([table_df, pd.DataFrame([min_row])], ignore_index=True)
        else:
            table_df = summary_df.round(4)
        output_components.extend([
            html.H4("Data Table", className="mt-3"),
            dash_table.DataTable(
                id={'type': 'abs-table', 'index': i},
                data=table_df.round(4).to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={
                    'textAlign': 'center',
                    'minWidth': '80px',
                    'maxWidth': '220px',
                    'width': '90px',
                    'overflow': 'hidden',
                    'textOverflow': 'ellipsis',
                    'whiteSpace': 'nowrap',
                },
                export_format='csv',
                export_headers='display',
                active_cell=None,
                row_selectable='single',
                cell_selectable=True,
            ),
            html.Div(id={'type': 'abs-run-drilldown', 'index': i}),
            # Stat-to-actual-key mapping summary
            html.Div([
                html.H6("Stat-to-actual-key mapping:"),
                html.Ul([
                    html.Li(f"{stat}: {', '.join(matched_keys.get(stat, [])) or 'None'}") for stat in stats_to_find
                ])
            ], style={"fontSize": "0.95em", "color": "#888", "marginTop": "10px"})
        ])
        # group별, stat별 그래프 (subgroup별 평균)
        if {'Subgroup', 'Stat', 'Value'}.issubset(summary_df.columns):
            stat_graphs = []
            for stat in stats_to_find:
                stat_df = summary_df.groupby(['Subgroup', 'Stat'])['Value'].mean().reset_index()
                stat_df = stat_df[stat_df['Stat'] == stat]
                fig = px.bar(stat_df, x='Subgroup', y='Value', title=f"{dir_name} - {stat} (Average)", 
                             labels={'Value': stat, 'Subgroup': 'Subgroup'},
                             color_discrete_sequence=[PLOTLY_COLORS[i % len(PLOTLY_COLORS)]])
                fig.update_layout(title_x=0.5, 
                                 plot_bgcolor=COLORS['white'],
                                 paper_bgcolor=COLORS['white'],
                                 font={'color': COLORS['dark']})
                stat_graphs.append(dbc.Col(dcc.Graph(figure=fig), width=12, lg=6, xl=4))
            output_components.append(dbc.Row(stat_graphs))
        output_components.append(html.Hr())
    return output_components

# --- 파스텔톤 히트맵 색상 함수 ---
def get_pastel_gradient_color(value, vmin, vmax, stat_name=None, color_direction=None):
    """
    파스텔 그라데이션 색상을 반환합니다.
    
    Args:
        value: 색상을 적용할 값
        vmin: 최소값
        vmax: 최대값
        stat_name: 통계 지표 이름 (예: 'ipc', 'power')
        color_direction: 색상 방향 ('green_for_positive' 또는 'red_for_positive')
    
    Returns:
        RGB 색상 문자열
    """
    if value is None or np.isnan(value):
        return 'white'
    
    # 색상 방향 결정
    if color_direction is None and stat_name is not None:
        color_direction = DEFAULT_COLOR_DIRECTIONS.get(stat_name, 'green_for_positive')
    elif color_direction is None:
        color_direction = 'green_for_positive'  # 기본값
    
    max_abs = max(abs(vmin), abs(vmax), 1e-6)
    ratio = min(abs(value) / max_abs, 1)
    
    if value > 0:
        if color_direction == 'green_for_positive':
            # 양수일 때 초록색 (기본): 나뭇잎 진녹색 파스텔
            r = int(220 - (160 * ratio))  # 220~60
            g = int(240 - (60 * ratio))   # 240~180
            b = int(220 - (130 * ratio))  # 220~90
        else:  # red_for_positive
            # 양수일 때 빨간색: 붉은 파스텔
            r = 255
            g = int(220 - (120 * ratio))  # 220~100
            b = int(220 - (120 * ratio))  # 220~100
        return f'rgb({r},{g},{b})'
    elif value < 0:
        if color_direction == 'green_for_positive':
            # 음수일 때 빨간색 (기본): 붉은 파스텔
            r = 255
            g = int(220 - (120 * ratio))  # 220~100
            b = int(220 - (120 * ratio))  # 220~100
        else:  # red_for_positive
            # 음수일 때 초록색: 나뭇잎 진녹색 파스텔
            r = int(220 - (160 * ratio))  # 220~60
            g = int(240 - (60 * ratio))   # 240~180
            b = int(220 - (130 * ratio))  # 220~90
        return f'rgb({r},{g},{b})'
    else:
        return 'white'

@app.callback(
    Output("percentage-change-tab", "children"),
    [Input("change-data-store", "data"),
     State("dir-names-store", "data"),
     State("alpha-slider", "value"),
     State("effect-size-threshold-slider", "value"),
     State('stats-store', 'data'),
     State('stat-matched-keys-store', 'data'),
     State('enable-statistical-options', 'value'),
     State('color-direction-store', 'data')]
)
def update_percentage_change_tab(change_dfs_json, dir_names, alpha, effect_size_threshold, stats_to_find, matched_keys, enable_statistical_options, color_directions):
    output_components = []
    if change_dfs_json is None:
        output_components.append("Please start analysis or wait for it to complete.")
        return output_components
    if not change_dfs_json:
        output_components.append("No comparison targets or analysis results.")
        return output_components
    baseline_dir_name = dir_names[0]
    for i, change_json in enumerate(change_dfs_json):
        target_dir_name = dir_names[i+1]
        comp_title = f"'{baseline_dir_name}' (Baseline) vs. '{target_dir_name}'"
        now_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_components.extend([
            html.Hr(className="my-4"),
            html.H2(f"🔍 Comparison: {comp_title}", className="mt-4")
        ])
        if change_json is None:
            output_components.append(html.P("Comparison failed for target directory."))
            continue
        table_df = pd.read_json(change_json, orient='split')
        # MultiIndex → 단일 인덱스(str)
        table_df.columns = [
            col if col == 'Subgroup' else f"{col[0]}_{col[1]}"
            for col in table_df.columns
        ]
        # stat-list 기준 컬럼 보장
        for stat in stats_to_find:
            if f"{stat}_Change" not in table_df.columns:
                table_df[f"{stat}_Change"] = np.nan
            if enable_statistical_options:
                if f"{stat}_p-adj" not in table_df.columns:
                    table_df[f"{stat}_p-adj"] = np.nan
                if f"{stat}_d" not in table_df.columns:
                    table_df[f"{stat}_d"] = np.nan
        # 컬럼 순서 맞추기
        columns = ['Subgroup']
        for stat in stats_to_find:
            columns.append(f"{stat}_Change")
            if enable_statistical_options:
                columns.extend([f"{stat}_p-adj", f"{stat}_d"])
        table_df = table_df[columns]
        stat_cols = [col for col in table_df.columns if col != 'Subgroup' and col.endswith('_Change')]
        style_data_conditional = []
        for stat_col in stat_cols:
            stat = stat_col.replace('_Change', '')
            vmin = table_df[stat_col].min()
            vmax = table_df[stat_col].max()
            for irow, row in table_df.iterrows():
                color = get_pastel_gradient_color(row[stat_col], vmin, vmax, stat, color_directions[stat])
                style_data_conditional.append({
                    'if': {'row_index': irow, 'column_id': stat_col},
                    'backgroundColor': color,
                    'color': 'black'
                })
        # 컬럼 정의
        dash_columns = []
        for stat in stats_to_find:
            dash_columns.append({"name": [stat, 'Change (%)'], "id": f"{stat}_Change", "presentation": "markdown"})
            if enable_statistical_options:
                dash_columns.append({"name": [stat, 'p-adj'], "id": f"{stat}_p-adj"})
                dash_columns.append({"name": [stat, 'd'], "id": f"{stat}_d"})
        dash_columns = [{"name": "Subgroup", "id": "Subgroup"}] + dash_columns
        # 데이터: 변화율+유의성 심벌(★)
        data = []
        for irow, row in table_df.iterrows():
            d = {}
            for col in table_df.columns:
                if col.endswith('_Change'):
                    val = row[col]
                    if enable_statistical_options:
                        padj_col = col.replace('_Change', '_p-adj')
                        d_col = col.replace('_Change', '_d')
                        pval = row[padj_col] if padj_col in table_df.columns else None
                        d_val = row[d_col] if d_col in table_df.columns else None
                        symbol = ''
                        if pd.notna(pval) and pval < alpha:
                            symbol = ' ★'
                        if pd.notna(val):
                            d[col] = f"{val:.2f}{symbol}"
                        else:
                            d[col] = ''
                    else:
                        d[col] = f"{val:.2f}" if pd.notna(val) else ''
                elif enable_statistical_options and col.endswith('_p-adj'):
                    val = row[col]
                    d[col] = f"{val:.4f}" if pd.notna(val) else ''
                elif enable_statistical_options and col.endswith('_d'):
                    val = row[col]
                    d[col] = f"{val:.4f}" if pd.notna(val) else ''
                else:
                    d[col] = row[col]
            data.append(d)
        output_components.extend([
            html.H4("Average Performance Change (%)", className="mt-3"),
            dash_table.DataTable(
                id={'type': 'change-table', 'index': i},
                data=data,
                columns=dash_columns,
                style_table={'overflowX': 'auto'},
                style_data_conditional=style_data_conditional,
                style_cell={'textAlign': 'center', 'minWidth': '80px', 'maxWidth': '100px', 'width': '90px'},
                row_selectable='single',
                cell_selectable=True,
                export_format='csv',
                export_headers='display',
            ),
            # 드릴다운(run-level) 그래프 컨테이너 추가
            html.Div(id={'type': 'run-level-graph', 'index': i}),
            # Stat-to-actual-key mapping summary
            html.Div([
                html.H6("Stat-to-actual-key mapping:"),
                html.Ul([
                    html.Li(f"{stat}: {', '.join(matched_keys.get(stat, [])) or 'None'}") for stat in stats_to_find
                ])
            ], style={"fontSize": "0.95em", "color": "#888", "marginTop": "10px"})
        ])
        # --- 그래프 추가 ---
        figures = []
        for stat in stats_to_find:
            stat_col = f"{stat}_Change"
            graph_df = table_df[['Subgroup', stat_col]].copy()
            graph_df = graph_df[graph_df['Subgroup'] != 'mean']
            graph_df = graph_df.dropna(subset=['Subgroup', stat_col])
            graph_df = graph_df[graph_df['Subgroup'] != '']
            graph_df = graph_df.reset_index(drop=True)
            if not graph_df.empty and len(graph_df['Subgroup']) == len(graph_df[stat_col]):
                fig = px.bar(graph_df, x='Subgroup', y=stat_col, title=f"{stat} Change (%)", 
                             labels={stat_col: 'Change (%)', 'Subgroup': 'Subgroup'},
                             color_discrete_sequence=[PLOTLY_COLORS[i % len(PLOTLY_COLORS)]])
                fig.update_layout(title_x=0.5,
                                 plot_bgcolor=COLORS['white'],
                                 paper_bgcolor=COLORS['white'],
                                 font={'color': COLORS['dark']})
                figures.append(dbc.Col(dcc.Graph(figure=fig), width=12, lg=6, xl=4))
        output_components.append(dbc.Row(figures))
    return output_components

# 드릴다운: row 클릭 시 run별 변화율 테이블(파스텔 히트맵 적용)
@app.callback(
    Output({'type': 'run-level-graph', 'index': MATCH}, 'children'),
    Input({'type': 'change-table', 'index': MATCH}, 'active_cell'),
    State({'type': 'change-table', 'index': MATCH}, 'data'),
    State('run-change-data-store', 'data'),
    State('dir-names-store', 'data'),
    State('color-direction-store', 'data'),
    prevent_initial_call=True,
)
def show_run_level_table(active_cell, table_data, run_change_dfs_json, dir_names, color_directions):
    if not active_cell or not table_data or not run_change_dfs_json:
        return []
    row = table_data[active_cell['row']]
    subgroup = row.get('Subgroup')
    if not subgroup:
        return []
    ctx = dash.callback_context
    idx = ctx.triggered[0]['prop_id'].split('.')[0]
    idx = eval(idx)['index'] if 'index' in idx else 0
    run_json = run_change_dfs_json[idx] if idx < len(run_change_dfs_json) else None
    if not run_json:
        return []
    run_df = pd.read_json(run_json, orient='split')
    run_df = run_df[run_df['Subgroup'] == subgroup]
    if run_df.empty:
        return html.P("No run-level data for this subgroup.")
    # wide format: Run, stat1, stat2, ...
    pivot_df = run_df.pivot_table(index='Run', columns='Stat', values='Change', aggfunc='mean').reset_index().round(2)
    # Run 자연스러운 오름차순 정렬
    pivot_df = pivot_df.sort_values('Run', key=lambda x: x.map(natural_key)).reset_index(drop=True)
    # 파스텔톤 히트맵 적용
    stat_cols = [col for col in pivot_df.columns if col != 'Run']
    style_data_conditional = []
    for stat in stat_cols:
        vmin = pivot_df[stat].min()
        vmax = pivot_df[stat].max()
        for irow, row in pivot_df.iterrows():
            color = get_pastel_gradient_color(row[stat], vmin, vmax, stat, color_directions[stat])
            style_data_conditional.append({
                'if': {'row_index': irow, 'column_id': stat},
                'backgroundColor': color,
                'color': 'black'
            })
    # 드릴다운(run-level) 테이블 하단에 평균(min) 행 추가
    min_row = {'Run': 'mean'}
    for stat in stat_cols:
        min_row[stat] = pivot_df[stat].mean()  # 평균, min으로 바꾸려면 .min()
    pivot_df = pd.concat([pivot_df, pd.DataFrame([min_row])], ignore_index=True)
    return dash_table.DataTable(
        data=pivot_df.round(4).to_dict('records'),
        columns=[{"name": c, "id": c} for c in pivot_df.columns],
        style_table={'overflowX': 'auto'},
        style_cell={'textAlign': 'center', 'minWidth': '80px', 'maxWidth': '100px', 'width': '90px'},
        style_data_conditional=style_data_conditional,
        page_size=20,
        export_format='csv',
        export_headers='display',
    )

# 콜백 6: [상호작용] '성능 변화율' 탭의 그룹 체크리스트에 따라 그래프를 업데이트
@app.callback(Output({'type': 'graph-container', 'index': MATCH}, 'children'), Input({'type': 'group-checklist', 'index': MATCH}, 'value'), State({'type': 'comparison-store', 'index': MATCH}, 'data'))
def update_graphs(selected_groups, json_data):
    if not selected_groups or not json_data: return []
    df = pd.read_json(json_data, orient='split')
    # change_df: columns=['Subgroup', 'L2_cache_miss_rate', 'ipc', ...]
    figures = []
    for stat in [col for col in df.columns if col != 'Subgroup']:
        fig = px.bar(df, x='Subgroup', y=stat, title=f"{stat} Change (%)", 
                     labels={stat: 'Change (%)', 'Subgroup': 'Subgroup'},
                     color_discrete_sequence=[PLOTLY_COLORS[i % len(PLOTLY_COLORS)]])
        fig.update_layout(title_x=0.5,
                         plot_bgcolor=COLORS['white'],
                         paper_bgcolor=COLORS['white'],
                         font={'color': COLORS['dark']})
        figures.append(dbc.Col(dcc.Graph(figure=fig), width=12, lg=6, xl=4))
    return dbc.Row(figures)

# 콜백 7: [상호작용] 'HTML로 저장' 버튼 클릭 시 파일 다운로드 실행
@app.callback(Output({'type': 'download-html', 'index': MATCH}, "data"), Input({'type': 'download-button', 'index': MATCH}, "n_clicks"), State({'type': 'comparison-store', 'index': MATCH}, "data"), State({'type': 'group-checklist', 'index': MATCH}, 'value'), State({'type': 'download-filename', 'index': MATCH}, 'value'), prevent_initial_call=True)
def download_html(n_clicks, json_data, selected_groups, filename):
    if not json_data or not selected_groups: return dash.no_update
    df = pd.read_json(json_data, orient='split')
    dff = df.loc[selected_groups].fillna(0).round(2).reset_index()
    html_string = dff.to_html(index=False, classes='table table-striped text-center', justify='center')
    final_filename = filename if filename else "report.html"
    if not final_filename.lower().endswith('.html'):
        final_filename += '.html'
    return dict(content=html_string, filename=final_filename)

# 콜백 8: [상호작용] '절대값 분석' 탭의 그룹 체크리스트에 따라 그래프를 업데이트
@app.callback(
    Output({'type': 'abs-graph-container', 'index': MATCH}, 'children'),
    Input({'type': 'abs-group-checklist', 'index': MATCH}, 'value'),
    State({'type': 'abs-summary-store', 'index': MATCH}, 'data'),
    prevent_initial_call=True,
)
def update_absolute_graphs(selected_groups, json_data):
    if not selected_groups or not json_data:
        return []
    df = pd.read_json(json_data, orient='split')
    # group, stat별로 그래프 생성
    figures = []
    for group in selected_groups:
        group_df = df[df['Group'] == group]
        for stat in group_df['Stat'].unique():
            stat_df = group_df[group_df['Stat'] == stat]
            fig = px.bar(stat_df, x='Subgroup', y='Value', title=f"{group} - {stat}", 
                         labels={'Value': stat, 'Subgroup': 'Subgroup'},
                         color_discrete_sequence=[PLOTLY_COLORS[i % len(PLOTLY_COLORS)]])
            fig.update_layout(title_x=0.5,
                             plot_bgcolor=COLORS['white'],
                             paper_bgcolor=COLORS['white'],
                             font={'color': COLORS['dark']})
            figures.append(dbc.Col(dcc.Graph(figure=fig), width=12, lg=6, xl=4))
    return dbc.Row(figures)

# 분석 시작 info 콜백
@app.callback(
    Output('analysis-info-start', 'children'),
    Input('run-button', 'n_clicks'),
    [State('baseline-dropdown', 'value'),
     State('target-checklist', 'value'),
     State('manual-path-input', 'value'),
     State('job-start-time-store', 'data')],
    prevent_initial_call=True,
)
def show_analysis_start_info(n_clicks, baseline_dir, target_dirs_checked, manual_paths, start_time):
    import datetime
    dirs = [baseline_dir] if baseline_dir else []
    if target_dirs_checked:
        dirs += list(target_dirs_checked)
    if manual_paths:
        dirs += [p.strip() for p in manual_paths.split('\n') if p.strip()]
    dirs = list(dict.fromkeys(dirs))
    if start_time:
        start_str = datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')
    else:
        start_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return [
        html.P(f"Target Directories: {', '.join(dirs) if dirs else 'None'}"),
        html.P(f"Analysis Start Time: {start_str}"),
        html.P("🔄 Analysis in progress...", style={"color": COLORS['secondary'], "fontWeight": "600", "fontStyle": "italic"})
    ]

# 분석 완료 info 콜백
@app.callback(
    Output('analysis-info-complete', 'children'),
    Input('analysis-complete-store', 'data'),
    State('job-start-time-store', 'data'),
    prevent_initial_call=True,
)
def show_analysis_complete_info(analysis_complete, start_time):
    if not analysis_complete:
        return ""
    import time
    elapsed = None
    if start_time:
        elapsed = time.time() - start_time
    msg = [html.P("✅ Analysis completed!", style={"color": COLORS['success'], "fontWeight": "bold", "fontSize": "1.1em"})]
    if elapsed is not None:
        mins, secs = divmod(int(elapsed), 60)
        msg.append(html.P(f"⏱️ Elapsed Time: {mins} min {secs} sec", style={"color": COLORS['info'], "fontWeight": "500"}))
    # 캐시 정보 추가
    cache_info = get_cache_info()
    if 'error' not in cache_info:
        msg.append(html.P(f"💾 Cache: {cache_info['cache_size']} files, "
                          f"{cache_info['hit_rate']:.1f}% hit rate",
                          style={"color": COLORS['secondary'], "fontSize": "0.9em"}))
    return msg

# 도움말 모달 제어 콜백
@app.callback(
    Output("help-modal", "is_open"),
    [Input("help-button", "n_clicks"),
     Input("close-help", "n_clicks")],
    [State("help-modal", "is_open")],
    prevent_initial_call=True,
)
def toggle_help_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

# HTML 저장 콜백 (JS 기반 안내)
@app.callback(
    Output("save-status", "children"),
    Input("save-html-btn", "n_clicks"),
    prevent_initial_call=True,
)
def save_dashboard_html(n_clicks):
    if not n_clicks:
        return ""
    return html.Span(
        "Dashboard snapshot saved as HTML (see toast notification at bottom right). Filename: dashboard_snapshot_YYYYMMDD_HHMMSS.html",
        style={"color": COLORS['info'], "fontWeight": "500"}
    )

# 사이드바 접힘/펼침 제어 콜백
@app.callback(
    [Output("sidebar-content", "is_open"),
     Output("sidebar-header", "is_open"),
     Output("sidebar-toggle", "children"),
     Output("sidebar-col", "width")],
    Input("sidebar-toggle", "n_clicks"),
    [State("sidebar-content", "is_open")],
    prevent_initial_call=True,
)
def toggle_sidebar(n_clicks, is_open):
    if n_clicks:
        if is_open:
            # 사이드바 접기
            return False, False, "▶", 1
        else:
            # 사이드바 펼치기
            return True, True, "◀", 3
    return is_open, True, "◀", 3

# --- study_out_dir 변경 시 하위 디렉토리 동적 업데이트 콜백 ---
@app.callback(
    [Output('baseline-dropdown', 'options'),
     Output('target-checklist', 'options'),
     Output('study-out-dir-status', 'children'),
     Output('study-out-dir-store', 'data')],
    Input('study-out-dir-search-btn', 'n_clicks'),
    State('study-out-dir-input', 'value'),
    prevent_initial_call=True,
)
def update_dir_options(n_clicks, study_out_dir):
    if not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    if not study_out_dir:
        return dash.no_update, dash.no_update, "Please enter a valid directory path.", dash.no_update
    
    try:
        discovered_dirs = find_subdirectories(Path(study_out_dir), depth=2)
        dir_options = [{'label': os.path.basename(p), 'value': p} for p in discovered_dirs]
        return dir_options, dir_options, f"✅ Found {len(discovered_dirs)} directories in {study_out_dir}", study_out_dir
    except Exception as e:
        return dash.no_update, dash.no_update, f"❌ Error: {str(e)}", dash.no_update

# 캐시 상태 확인 함수
def get_cache_info():
    """
    캐시 상태 정보를 반환
    """
    try:
        cache_stats = cache.stats()
        cache_size = len(cache)
        cache_volume = cache.volume()
        
        # stats가 tuple인 경우 처리
        if isinstance(cache_stats, dict):
            hits = cache_stats.get('hits', 0)
            misses = cache_stats.get('misses', 0)
        else:
            # stats가 tuple인 경우 (hits, misses)
            hits, misses = cache_stats if len(cache_stats) >= 2 else (0, 0)
        
        hit_rate = hits / max(hits + misses, 1) * 100
        
        return {
            'cache_hits': hits,
            'cache_misses': misses,
            'cache_size': cache_size,
            'cache_volume': cache_volume,
            'cache_directory': cache_directory,
            'hit_rate': hit_rate
        }
    except Exception as e:
        return {'error': str(e)}

# 1. subgroup별 샘플 파일 선정 및 offset 캐싱
def build_all_offsets(report_files, stats_to_find, subgroups):
    # {subgroup: offsets}
    offsets_dict = {}
    for subgroup in subgroups:
        # 해당 subgroup의 첫 번째 파일을 샘플로 사용
        for file in report_files:
            subgroup_name = Path(file).parent.parent.name
            if subgroup in subgroup_name:
                offsets = build_offsets_for_subgroup(subgroup, file, stats_to_find)
                if offsets:
                    offsets_dict[subgroup] = offsets
                break
    return offsets_dict

# 2. process_single_report 수정
def process_single_report_mmap(report_path: Path, stats_to_find: list, group_name: str, offsets_dict, subgroups):
    run_name = report_path.parent.name
    subgroup_name = report_path.parent.parent.name
    for mon_test in subgroups:
        if mon_test in subgroup_name:
            subgroup_name = mon_test
    offsets = offsets_dict.get(subgroup_name)
    if not offsets:
        return None
    values_dict = parse_file_with_offsets(str(report_path), offsets, stats_to_find)
    if not values_dict:
        return None
    averages = {stat: v for stat, v in values_dict.items() if v is not None}
    matched_keys_dict = {stat: [stat] for stat in averages}
    result = (group_name, subgroup_name, run_name, averages, matched_keys_dict) if averages else None
    return result

# 3. analyze_directory 수정
def analyze_directory(root_path: Path, stats_to_find: list):
    report_files = list(root_path.rglob("*.out"))
    if not report_files:
        return None, {}
    group_name = root_path.name
    subgroups = ['sub1','sub2','sub3','sub4']  # 하드코딩된 subgroups
    offsets_dict = build_all_offsets(report_files, stats_to_find, subgroups)
    results = []
    all_matched_keys = {stat: set() for stat in stats_to_find}
    # stats_hash = hashlib.md5(','.join(stats_to_find).encode()).hexdigest()
    # uncached_files = []
    # cache_results = {}
    # for report_path in report_files:
    #     cache_key = (str(report_path.resolve()), os.path.getmtime(report_path), stats_hash)
    #     cached = cache.get(cache_key, default=None)
    #     if cached is not None:
    #         if isinstance(cached, tuple) and len(cached) == 4:
    #             group, subgroup, run, averages = cached
    #             matched_keys_dict = {}
    #             cache_results[report_path] = (group, subgroup, run, averages, matched_keys_dict)
    #         else:
    #             cache_results[report_path] = cached
    #     else:
    #         uncached_files.append(report_path)
    # if uncached_files:
    #     with ProcessPoolExecutor(max_workers=8) as executor:
    #         futures = [
    #             executor.submit(process_single_report_mmap, file, stats_to_find, group_name, offsets_dict, subgroups)
    #             for file in uncached_files
    #         ]
    #         for idx, future in enumerate(as_completed(futures)):
    #             result = future.result()
    #             if result:
    #                 report_path = uncached_files[idx]
    #                 cache_key = (str(report_path.resolve()), os.path.getmtime(report_path), stats_hash)
    #                 cache.set(cache_key, result)
    #                 cache_results[report_path] = result
    # for report_path in report_files:
    #     result = cache_results.get(report_path)
    #     if result:
    #         group, subgroup, run, stats, matched_keys_dict = result
    #         results.append((group, subgroup, run, stats))
    #         for stat, keys in matched_keys_dict.items():
    #             all_matched_keys[stat].update(keys)
    
    # 캐시 비활성화: 모든 파일을 새로 처리
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_single_report_mmap, file, stats_to_find, group_name, offsets_dict, subgroups)
            for file in report_files
        ]
        for future in as_completed(futures):
            result = future.result()
            if result:
                group, subgroup, run, stats, matched_keys_dict = result
                results.append((group, subgroup, run, stats))
                for stat, keys in matched_keys_dict.items():
                    all_matched_keys[stat].update(keys)
    
    if not results:
        return None, {stat: list(keys) for stat, keys in all_matched_keys.items()}
    records = []
    for group, subgroup, run, stats in results:
        for stat, value in stats.items():
            records.append({
                'Group': group,
                'Subgroup': subgroup,
                'Run': run,
                'Stat': stat,
                'Value': value
            })
    df = pd.DataFrame(records)
    if df.empty:
        return None, {stat: list(keys) for stat, keys in all_matched_keys.items()}
    df['SubgroupPrefix'] = df['Subgroup'].str[:5]
    df_agg = df.groupby(['Group', 'SubgroupPrefix', 'Run', 'Stat'], as_index=False)['Value'].mean()
    df_agg = df_agg.rename(columns={'SubgroupPrefix': 'Subgroup'})
    return df_agg, {stat: list(keys) for stat, keys in all_matched_keys.items()}

def build_offsets_for_subgroup(subgroup, sample_file, stats_to_find):
    """
    sample_file: 해당 subgroup의 대표 파일 경로 (Path 또는 str)
    stats_to_find: stat 리스트
    return: {stat: offset_from_running_complete}
    """
    offsets = {}
    with open(sample_file, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        # Running Complete 위치 찾기
        running_idx = next((i for i, line in enumerate(lines) if 'Running Complete' in line), None)
        if running_idx is None:
            return None
        for stat in stats_to_find:
            for offset, line in enumerate(lines[running_idx+1:], 1):
                if stat in line:
                    offsets[stat] = offset
                    break
    return offsets

def parse_file_with_offsets(file_path, offsets, stats_to_find):
    """
    file_path: 분석할 파일 경로 (str)
    offsets: {stat: offset_from_running_complete}
    stats_to_find: stat 리스트
    return: {stat: value}
    """
    results = {}
    failed_stats = set()
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        mm.seek(0)
        pos = mm.find(b'Running Complete')
        if pos == -1:
            mm.close()
            return None
        mm.seek(pos)
        mm.readline()  # Running Complete 라인 넘기기
        for stat in stats_to_find:
            offset = offsets.get(stat)
            if offset is None:
                failed_stats.add(stat)
                continue
            mm.seek(pos)
            mm.readline()  # Running Complete 라인
            for _ in range(offset):
                mm.readline()
            line = mm.readline().decode('utf-8', errors='ignore')
            match = KEY_VALUE_PATTERN.search(line)
            if match:
                _, value_str = match.groups()
                results[stat] = float(value_str) if value_str.lower() != 'nan' else 0.0
            else:
                failed_stats.add(stat)
        mm.close()
    # Fallback: offset 방식에서 못 찾은 stat은 line by line으로 찾기
    if failed_stats:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            search_started = False
            for line in f:
                if not search_started:
                    if "Running Complete" in line:
                        search_started = True
                    continue
                match = KEY_VALUE_PATTERN.search(line)
                if match:
                    found_key, value_str = match.groups()
                    for stat in failed_stats.copy():
                        if stat in found_key:
                            results[stat] = 0.0 if value_str.lower() == 'nan' else float(value_str)
                            failed_stats.remove(stat)
                if not failed_stats:
                    break
    return results

# [드릴다운] 절대값 분석 테이블 셀 클릭 시 run별 절대값 테이블 표시
@app.callback(
    Output({'type': 'abs-run-drilldown', 'index': MATCH}, 'children'),
    Input({'type': 'abs-table', 'index': MATCH}, 'active_cell'),
    State({'type': 'abs-table', 'index': MATCH}, 'data'),
    State('summary-data-store', 'data'),
    State('dir-names-store', 'data'),
    State('stats-store', 'data'),
    State('color-direction-store', 'data'),
    prevent_initial_call=True,
)
def show_abs_run_level_table(active_cell, table_data, summaries_json, dir_names, stats_to_find, color_directions):
    if not active_cell or not table_data or not summaries_json or not dir_names:
        return []
    row = table_data[active_cell['row']]
    col_id = active_cell['column_id']
    # column_id가 dict일 경우 str로 변환
    if isinstance(col_id, dict):
        col_id = col_id.get('id', '')
    subgroup = row.get('Subgroup')
    # mean 행 클릭 시 무시
    if not subgroup or subgroup == 'mean':
        return []
    # triggered prop_id에서 index 안전 추출
    ctx = dash.callback_context
    idx = None
    try:
        prop_id = ctx.triggered[0]['prop_id'].split('.')[0]
        if 'index' in prop_id:
            idx = int(eval(prop_id)['index'])
    except Exception:
        idx = 0
    if idx is None or idx >= len(summaries_json):
        return []
    summary_json = summaries_json[idx]
    if not summary_json:
        return []
    summary_df = pd.read_json(summary_json, orient='split')
    # run별, stat별 wide format
    if {'Subgroup', 'Run', 'Stat', 'Value'}.issubset(summary_df.columns):
        run_df = summary_df[summary_df['Subgroup'] == subgroup]
        if run_df.empty:
            return html.P("No run-level data for this subgroup.")
        pivot_df = run_df.pivot_table(index='Run', columns='Stat', values='Value', aggfunc='mean').reset_index().round(4)
        # stat-list 기준 컬럼 보장
        for stat in stats_to_find:
            if stat not in pivot_df.columns:
                pivot_df[stat] = np.nan
        pivot_df = pivot_df[['Run'] + stats_to_find]
        # Run 자연스러운 오름차순 정렬
        pivot_df = pivot_df.sort_values('Run', key=lambda x: x.map(natural_key)).reset_index(drop=True)
        # 평균 행 추가
        min_row = {'Run': 'mean'}
        for stat in stats_to_find:
            min_row[stat] = pivot_df[stat].mean()
        pivot_df = pd.concat([pivot_df, pd.DataFrame([min_row])], ignore_index=True)
        return dash_table.DataTable(
            data=pivot_df.round(4).to_dict('records'),
            columns=[{"name": c, "id": c} for c in pivot_df.columns],
            style_table={'overflowX': 'auto'},
            style_cell={
                'textAlign': 'center',
                'minWidth': '80px',
                'maxWidth': '220px',
                'width': '90px',
                'overflow': 'hidden',
                'textOverflow': 'ellipsis',
                'whiteSpace': 'nowrap',
            },
            page_size=20,
            export_format='csv',
            export_headers='display',
        )
    return html.P("No run-level data for this subgroup.")

# --- 새로운 탭: Detailed Run Table ---

detailed_tab_id = 'detailed-run-tab'

def get_group_subgroup_options(summaries_json, dir_names):
    # summaries_json: [baseline, target1, ...]
    # dir_names: [baseline, target1, ...]
    options = []
    if not summaries_json or not dir_names:
        return options
    for i, summary_json in enumerate(summaries_json):
        if not summary_json:
            continue
        df = pd.read_json(summary_json, orient='split')
        if {'Group', 'Subgroup'}.issubset(df.columns):
            for _, row in df[['Group', 'Subgroup']].drop_duplicates().iterrows():
                label = f"{row['Group']} / {row['Subgroup']}"
                value = f"{i}|||{row['Group']}|||{row['Subgroup']}"
                options.append({'label': label, 'value': value})
    return options

@app.callback(
    Output('detailed-group-subgroup-dropdown', 'options'),
    [Input('summary-data-store', 'data'), State('dir-names-store', 'data')]
)
def update_detailed_group_subgroup_options(summaries_json, dir_names):
    return get_group_subgroup_options(summaries_json, dir_names)

@app.callback(
    Output('detailed-run-table-container', 'children'),
    [Input('detailed-group-subgroup-dropdown', 'value'),
     State('summary-data-store', 'data'),
     State('run-change-data-store', 'data'),
     State('dir-names-store', 'data'),
     State('stats-store', 'data'),
     State('color-direction-store', 'data')]
)
def update_detailed_run_table(selected_value, summaries_json, run_change_dfs_json, dir_names, stats_to_find, color_directions):
    if not selected_value or not summaries_json or not dir_names or not stats_to_find:
        return "Please select a group/subgroup."
    # value: "i|||Group|||Subgroup"
    i, group, subgroup = selected_value.split('|||')
    i = int(i)
    summary_json = summaries_json[i]
    if not summary_json:
        return "No data."
    summary_df = pd.read_json(summary_json, orient='split')
    # 해당 group/subgroup의 run별 데이터
    run_df = summary_df[(summary_df['Group'] == group) & (summary_df['Subgroup'] == subgroup)]
    if run_df.empty:
        return "No run-level data for this group/subgroup."
    # run별, stat별 절대값
    abs_pivot = run_df.pivot_table(index='Run', columns='Stat', values='Value', aggfunc='mean').reset_index()
    # Run 자연스러운 오름차순 정렬
    abs_pivot = abs_pivot.sort_values('Run', key=lambda x: x.map(natural_key)).reset_index(drop=True)
    
    # run_change_data_store에서 해당 group/subgroup의 run별 변화량 데이터 가져오기
    change_pivot = None
    if run_change_dfs_json and i > 0 and i-1 < len(run_change_dfs_json) and run_change_dfs_json[i-1]:
        run_change_df = pd.read_json(run_change_dfs_json[i-1], orient='split')
        # 해당 subgroup의 run별 변화량 데이터 필터링
        subgroup_change_df = run_change_df[run_change_df['Subgroup'] == subgroup]
        if not subgroup_change_df.empty:
            # run별, stat별 변화량 pivot
            change_pivot = subgroup_change_df.pivot_table(index='Run', columns='Stat', values='Change', aggfunc='mean').reset_index()
            change_pivot = change_pivot.sort_values('Run', key=lambda x: x.map(natural_key)).reset_index(drop=True)
    
    # baseline 비교를 위해 baseline summary도 준비 (run별 변화량이 없을 때 사용)
    baseline_df = None
    if i != 0 and summaries_json[0]:
        baseline_df = pd.read_json(summaries_json[0], orient='split')
    
    # 변화량 계산
    if change_pivot is not None:
        # run_change_data_store에서 가져온 변화량 사용
        merged = abs_pivot.merge(change_pivot, on='Run', how='left', suffixes=('', '_Change'))
        # 컬럼명 정리
        for stat in stats_to_find:
            if f'{stat}_Change' in merged.columns:
                merged[f'{stat}_Change'] = merged[f'{stat}_Change']
            else:
                merged[f'{stat}_Change'] = np.nan
        abs_pivot = merged
    elif baseline_df is not None:
        # baseline과 직접 계산
        base_df = baseline_df[(baseline_df['Group'] == group) & (baseline_df['Subgroup'] == subgroup)]
        if not base_df.empty:
            base_pivot = base_df.pivot_table(index='Run', columns='Stat', values='Value', aggfunc='mean').reset_index()
            base_pivot = base_pivot.sort_values('Run', key=lambda x: x.map(natural_key)).reset_index(drop=True)
            # run 이름 기준 merge (left: target, right: baseline)
            merged = abs_pivot.merge(base_pivot, on='Run', how='left', suffixes=('', '_baseline'))
            # 변화량 계산
            for stat in stats_to_find:
                merged[f'{stat}_Change'] = np.nan  # 기본값
                for idx, row in merged.iterrows():
                    target_val = row.get(stat)
                    baseline_val = row.get(f'{stat}_baseline')
                    if pd.notna(target_val) and pd.notna(baseline_val) and baseline_val != 0:
                        merged.at[idx, f'{stat}_Change'] = ((target_val - baseline_val) / baseline_val) * 100
            abs_pivot = merged  # 이후 열 가공에 사용
        else:
            # baseline에 해당 group/subgroup이 없으면 모든 변화량을 NaN으로
            for stat in stats_to_find:
                abs_pivot[f'{stat}_Change'] = np.nan
    else:
        # baseline이 없으면 모든 변화량을 NaN으로
        for stat in stats_to_find:
            abs_pivot[f'{stat}_Change'] = np.nan
    # columns: Run, stat1_Absolute, stat1_Change, stat2_Absolute, stat2_Change, ...
    columns = [{'name': 'Run', 'id': 'Run'}]
    data = []
    for stat in stats_to_find:
        columns.append({'name': [stat, 'Absolute'], 'id': stat})
        columns.append({'name': [stat, 'Change(%)'], 'id': f'{stat}_Change'})
    # 데이터 가공
    for _, row in abs_pivot.iterrows():
        d = {'Run': row['Run']}
        for stat in stats_to_find:
            val = row.get(stat, np.nan)
            chg_col = f'{stat}_Change'
            chg = row.get(chg_col, np.nan) if chg_col in abs_pivot.columns else np.nan
            d[stat] = round(val, 2) if pd.notna(val) else val
            d[chg_col] = round(chg, 2) if pd.notna(chg) else chg
        data.append(d)
    # 변화량 cell에만 히트맵 적용
    style_data_conditional = []
    for stat in stats_to_find:
        col = f'{stat}_Change'
        if col in abs_pivot.columns:
            vmin = abs_pivot[col].min()
            vmax = abs_pivot[col].max()
            for irow, prow in abs_pivot.iterrows():
                val = prow.get(col, None)
                color_direction = color_directions.get(stat, 'green_for_positive') if color_directions else 'green_for_positive'
                color = get_pastel_gradient_color(val, vmin, vmax, stat, color_direction) if vmin is not None and vmax is not None else 'white'
                style_data_conditional.append({
                    'if': {'row_index': irow, 'column_id': col},
                    'backgroundColor': color,
                    'color': 'black'
                })
    return dash_table.DataTable(
        data=data,
        columns=columns,
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'center',
            'minWidth': '80px',
            'maxWidth': '220px',
            'width': '90px',
            'overflow': 'hidden',
            'textOverflow': 'ellipsis',
            'whiteSpace': 'nowrap',
        },
        style_data_conditional=style_data_conditional,
        merge_duplicate_headers=True,
        export_format='csv',
        export_headers='display',
    )

# 색상 방향 설정을 위한 콜백 함수들
@app.callback(
    Output('color-direction-controls', 'children'),
    Input('stats-store', 'data')
)
def update_color_direction_controls(stats_to_find):
    """통계 지표 목록에 따라 색상 방향 설정 컨트롤을 동적으로 생성"""
    if not stats_to_find:
        return []
    
    controls = []
    for stat in stats_to_find:
        default_direction = DEFAULT_COLOR_DIRECTIONS.get(stat, 'green_for_positive')
        controls.append(
            html.Div([
                html.Label(f"{stat}:", style={"fontWeight": "600", "marginBottom": "5px"}),
                dcc.RadioItems(
                    id={'type': 'color-direction', 'stat': stat},
                    options=[
                        {'label': '🔴 Red for +', 'value': 'red_for_positive'},
                        {'label': '🟢 Green for +', 'value': 'green_for_positive'}
                    ],
                    value=default_direction,
                    inline=True,
                    style={"fontSize": "0.9em"}
                )
            ], style={"marginBottom": "10px"})
        )
    return controls

@app.callback(
    [Output('color-direction-store', 'data'),
     Output('color-direction-status', 'children')],
    [Input('apply-colors-btn', 'n_clicks')] + 
    [Input({'type': 'color-direction', 'stat': ALL}, 'value')],
    [State({'type': 'color-direction', 'stat': ALL}, 'id'),
     State('stats-store', 'data')],
    prevent_initial_call=True
)
def update_color_directions(n_clicks, color_values, color_ids, stats_to_find):
    """색상 방향 설정을 저장하고 상태를 업데이트"""
    if not n_clicks or not stats_to_find:
        return dash.no_update, dash.no_update
    
    # 새로운 색상 방향 설정 구성
    new_color_directions = DEFAULT_COLOR_DIRECTIONS.copy()
    
    # 각 통계 지표의 색상 방향 업데이트
    for i, color_id in enumerate(color_ids):
        if i < len(color_values) and color_values[i]:
            stat_name = color_id['stat']
            new_color_directions[stat_name] = color_values[i]
    
    # stats_to_find에 있는 모든 지표에 대해 기본값 설정
    for stat in stats_to_find:
        if stat not in new_color_directions:
            new_color_directions[stat] = 'green_for_positive'
    
    return new_color_directions, f"✅ Color directions updated for {len(stats_to_find)} stats"

def split_by_last_dot(text):
    """마지막 . 을 기준으로 텍스트를 분리합니다."""
    if '.' not in text:
        return "", text
    
    # 마지막 . 의 위치를 찾습니다
    last_dot_index = text.rfind('.')
    
    # 마지막 . 을 기준으로 앞뒤를 분리합니다
    prefix = text[:last_dot_index]
    suffix = text[last_dot_index + 1:]  # +1로 . 을 제거합니다
    
    return prefix, suffix

if __name__ == "__main__":
    app.run(debug=False)
