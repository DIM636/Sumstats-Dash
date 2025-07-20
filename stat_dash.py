#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, dash_table, Input, Output, State, MATCH, ALL, DiskcacheManager
import pandas as pd
import plotly.express as px
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time
import diskcache
import datetime
import numpy as np
import sys
import dash
import plotly

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

def process_single_report(report_path: Path, stats_to_find: list):
    # [설명] 단일 결과 파일에서 stat 값을 파싱하여 반환합니다. 캐시를 활용해 중복 파싱을 방지합니다.
    # 파일별 캐시 키: (경로, mtime)
    cache_key = (str(report_path.resolve()), os.path.getmtime(report_path))
    cached = cache.get(cache_key, default=None)
    if cached is not None:
        return cached
    # group, subgroup, run 추출
    run_name = report_path.parent.name
    subgroup_name = report_path.parent.parent.name
    group_name = report_path.parent.parent.parent.name
    values_dict = {stat: [] for stat in stats_to_find}
    search_started = False
    try:
        with open(report_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if not search_started:
                    if "Running Complete" in line:
                        search_started = True
                    continue
                match = KEY_VALUE_PATTERN.search(line)
                if match:
                    found_key, value_str = match.groups()
                    if found_key in stats_to_find:
                        value = 0.0 if value_str.lower() == 'nan' else float(value_str)
                        values_dict[found_key].append(value)
    except IOError:
        return None
    averages = {stat: sum(v) / len(v) for stat, v in values_dict.items() if v}
    result = (group_name, subgroup_name, run_name, averages) if averages else None
    cache.set(cache_key, result)
    return result


def analyze_directory(root_path: Path, stats_to_find: list):
    # [설명] 지정한 디렉토리 내 모든 결과 파일을 병렬로 분석하여 DataFrame으로 집계합니다.
    report_files = list(root_path.rglob("*.out"))
    if not report_files:
        return None
    results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_single_report, file, stats_to_find) for file in report_files]
        for future in as_completed(futures):
            if result := future.result():
                results.append(result)
    if not results:
        return None
    # group, subgroup, run, stat별로 DataFrame 생성
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
        return None
    # 항상 Run 컬럼 포함해서 반환
    return df


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
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], background_callback_manager=background_callback_manager)
server = app.server
app.title = "시뮬레이션 분석기"

# --- 앱 레이아웃 ---
# [설명] 대시보드의 전체 UI 구조(탭, 버튼, 설명, 데이터 저장소 등)를 정의합니다.
discovered_dirs = find_subdirectories(Path.cwd(), depth=2)
dir_options = [{'label': os.path.basename(p), 'value': p} for p in discovered_dirs]

controls = dbc.Card(dbc.CardBody([
    html.H4("1. Select Baseline Directory", className="card-title"),
    html.Div([
        html.P("This tool analyzes the following stats:", style={"fontWeight": "bold"}),
        html.Ul([html.Li(stat) for stat in STATS_TO_FIND]),
        html.P("Click 'Start Analysis' to automatically aggregate and compare the above stats for the selected directories.", style={"fontSize": "0.95em", "color": "#555"})
    ], className="mb-3"),
    dcc.Dropdown(id='baseline-dropdown', options=dir_options, placeholder="Select the baseline directory..."),
    html.Hr(className="my-3"),
    html.H4("2. Select Target Directories"),
    dcc.Checklist(id='target-checklist', options=dir_options, labelClassName="me-3", inputClassName="me-1"),
    html.Hr(className="my-3"),
    html.H4("3. (Optional) Add Directories Manually"),
    dbc.Textarea(id="manual-path-input", placeholder="Enter directory paths not listed above, one per line...", style={'height': '80px'}),
    dbc.Button("Start Analysis", id="run-button", color="primary", className="my-3 w-100"),
    html.Div(id="progress-wrapper", children=[dbc.Progress(id="progress-bar", value=100, striped=True, animated=True)], style={'display': 'none'}),
    html.Div(id="analysis-info", className="mt-3"),
]))

# --- percentage-change-tab 상단 옵션 UI 추가 ---
stat_options = [{'label': stat, 'value': stat} for stat in STATS_TO_FIND]

app.layout = dbc.Container([
    html.H1("📊 Monaco Simulation Results Analysis & Comparison", className="my-4"),
    html.Button("Save as HTML", id="save-html-btn", className="mb-3"),
    dbc.Tabs(id="tabs-container", children=[
        dbc.Tab(controls, label="1. Analysis Setup"),
        dbc.Tab(html.Div(id="absolute-values-tab"), label="2. Absolute Value Analysis"),
        dbc.Tab(html.Div(id="percentage-change-tab"), label="3. Performance Change Analysis"),
    ]),
    dcc.Store(id='job-start-time-store'),
    dcc.Store(id='summary-data-store'),
    dcc.Store(id='change-data-store'),
    dcc.Store(id='dir-names-store'),
    dcc.Store(id='run-change-data-store'),
    dcc.Store(id='analysis-meta-store'),
    dcc.Interval(id='progress-interval', interval=1000, disabled=True),
    html.Footer([
        html.Hr(),
        html.Div(
            [
                html.Span("Sumstats Dash — Created by dong63.ma", style={"marginRight": "10px"}),
                html.Span("| Version: v1.0.0", style={"marginRight": "10px"}),
                html.Span("| Last updated: 2025-07-20", style={"marginRight": "10px"}),
                html.Span(f"| Python {sys.version_info.major}.{sys.version_info.minor} | Dash {dash.__version__} | Plotly {plotly.__version__}")
            ],
            style={"textAlign": "center", "color": "#888", "fontSize": "0.95em", "margin": "20px 0"}
        )
    ])
], fluid=True)

# --- 콜백 함수들 ---
# [설명] 사용자 인터랙션(분석 시작, 진행/완료 표시, 데이터 분석, 시각화 등)에 따라 동적으로 UI를 업데이트하는 Dash 콜백 함수들입니다.

# 콜백 1: '분석 시작' 버튼 클릭 시, 스토어 초기화, 진행률 바 활성화, 버튼 비활성화
@app.callback(
    [Output('job-start-time-store', 'data'),
     Output('progress-interval', 'disabled'),
     Output('progress-wrapper', 'style'),
     Output('run-button', 'disabled'),
     Output('summary-data-store', 'data', allow_duplicate=True),
     Output('change-data-store', 'data', allow_duplicate=True),
     Output('dir-names-store', 'data', allow_duplicate=True)],
    Input('run-button', 'n_clicks'),
    prevent_initial_call=True,
)
def start_analysis_job(n_clicks):
    start_time = time.time()
    return start_time, False, {'display': 'block'}, True, None, None, None

# 콜백 2: [백그라운드] 실제 분석을 수행하고 결과를 dcc.Store에 저장
@app.callback(
    [Output('summary-data-store', 'data'),
     Output('change-data-store', 'data'),
     Output('dir-names-store', 'data'),
     Output('run-change-data-store', 'data')],
    Input('run-button', 'n_clicks'),
    [State("baseline-dropdown", "value"),
     State("target-checklist", "value"),
     State("manual-path-input", "value")],
    background=True,
    manager=background_callback_manager,
    prevent_initial_call=True,
)
def run_analysis_callback(set_progress, baseline_dir, target_dirs_checked, manual_paths):
    if not baseline_dir: return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    targets = set(target_dirs_checked) if target_dirs_checked else set()
    if manual_paths:
        for path in manual_paths.split('\n'):
            if path.strip(): targets.add(path.strip())
    targets.discard(baseline_dir)
    if not targets: return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    root_dirs = [baseline_dir] + sorted(list(targets))
    dir_names = [os.path.basename(p) for p in root_dirs]
    summaries = [analyze_directory(Path(dir_path), STATS_TO_FIND) for dir_path in root_dirs]
    
    summaries_json = [s.to_json(orient='split') if s is not None else None for s in summaries]
    change_dfs_json = []
    run_change_dfs_json = []
    if summaries[0] is not None:
        for i in range(1, len(summaries)):
            if summaries[i] is not None:
                # 평균 집계
                base = summaries[0].groupby(['Subgroup', 'Stat'])['Value'].mean().reset_index()
                targ = summaries[i].groupby(['Subgroup', 'Stat'])['Value'].mean().reset_index()
                merged = pd.merge(targ, base, on=['Subgroup', 'Stat'], suffixes=('_target', '_baseline'))
                merged['Change'] = ((merged['Value_target'] - merged['Value_baseline']) / merged['Value_baseline']) * 100
                pivot_df = merged.pivot_table(index='Subgroup', columns='Stat', values='Change', aggfunc='mean').reset_index()
                table_df = pivot_df.round(2).fillna(0)
                change_dfs_json.append(table_df.to_json(orient='split'))
                # run별 변화율 계산
                base_run = summaries[0][['Subgroup', 'Run', 'Stat', 'Value']].copy()
                targ_run = summaries[i][['Subgroup', 'Run', 'Stat', 'Value']].copy()
                run_merged = pd.merge(targ_run, base_run, on=['Subgroup', 'Run', 'Stat'], suffixes=('_target', '_baseline'))
                run_merged['Change'] = ((run_merged['Value_target'] - run_merged['Value_baseline']) / run_merged['Value_baseline']) * 100
                run_change_dfs_json.append(run_merged.to_json(orient='split'))
            else:
                change_dfs_json.append(None)
                run_change_dfs_json.append(None)
    return summaries_json, change_dfs_json, dir_names, run_change_dfs_json

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
    Input("summary-data-store", "data"),
    State("dir-names-store", "data"),
)
def update_absolute_values_tab(summaries_json, dir_names):
    if summaries_json is None:
        return "Please start analysis or wait for it to complete."
    output_components = []
    for i, summary_json in enumerate(summaries_json):
        if summary_json is None:
            continue
        dir_name = dir_names[i]
        summary_df = pd.read_json(summary_json, orient='split')
        output_components.append(html.H3(f"📁 {dir_name} Absolute Value Analysis", className="mt-4"))
        # subgroup별 stat별 평균 테이블
        if {'Subgroup', 'Stat', 'Value'}.issubset(summary_df.columns):
            pivot_df = summary_df.groupby(['Subgroup', 'Stat'])['Value'].mean().reset_index()
            table_df = pivot_df.pivot(index='Subgroup', columns='Stat', values='Value').reset_index().round(4)
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
                data=table_df.round(4).to_dict('records'),
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'center', 'minWidth': '80px', 'maxWidth': '100px', 'width': '90px'},
                export_format='csv',
                export_headers='display',
            )
        ])
        # group별, stat별 그래프 (subgroup별 평균)
        if {'Subgroup', 'Stat', 'Value'}.issubset(summary_df.columns):
            stat_graphs = []
            for stat in summary_df['Stat'].unique():
                stat_df = summary_df.groupby(['Subgroup', 'Stat'])['Value'].mean().reset_index()
                stat_df = stat_df[stat_df['Stat'] == stat]
                fig = px.bar(stat_df, x='Subgroup', y='Value', title=f"{dir_name} - {stat} (Average)", labels={'Value': stat, 'Subgroup': 'Subgroup'})
                fig.update_layout(title_x=0.5)
                stat_graphs.append(dbc.Col(dcc.Graph(figure=fig), width=12, lg=6, xl=4))
            output_components.append(dbc.Row(stat_graphs))
        output_components.append(html.Hr())
    return output_components

# --- 파스텔톤 히트맵 색상 함수 ---
def get_pastel_gradient_color(value, vmin, vmax):
    if value is None or np.isnan(value):
        return 'white'
    max_abs = max(abs(vmin), abs(vmax), 1e-6)
    ratio = min(abs(value) / max_abs, 1)
    if value > 0:
        # 나뭇잎 진녹색 파스텔: 밝은 연녹색(220,240,220) ~ 진녹색(60,180,90)
        r = int(220 - (160 * ratio))  # 220~60
        g = int(240 - (60 * ratio))   # 240~180
        b = int(220 - (130 * ratio))  # 220~90
        return f'rgb({r},{g},{b})'
    elif value < 0:
        # 붉은 파스텔: 밝은 연분홍(255,220,220) ~ 진빨강(255,100,100)
        r = 255
        g = int(220 - (120 * ratio))  # 220~100
        b = int(220 - (120 * ratio))  # 220~100
        return f'rgb({r},{g},{b})'
    else:
        return 'white'

@app.callback(
    Output("percentage-change-tab", "children"),
    Input("change-data-store", "data"),
    State("dir-names-store", "data"),
)
def update_percentage_change_tab(change_dfs_json, dir_names):
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
        output_components.extend([
            html.Hr(className="my-4"),
            html.H2(f"🔍 Comparison: {comp_title}", className="mt-4")
        ])
        if change_json is None:
            output_components.append(html.P("Comparison failed for target directory."))
            continue
        change_df = pd.read_json(change_json, orient='split')
        stat_cols = [col for col in change_df.columns if col != 'Subgroup']
        table_df = change_df[['Subgroup'] + stat_cols].copy()
        # 평균(min) 행 추가
        min_row = {'Subgroup': 'mean'}
        for stat in stat_cols:
            min_row[stat] = table_df[stat].mean()  # 평균, min으로 바꾸려면 .min()
        table_df = pd.concat([table_df, pd.DataFrame([min_row])], ignore_index=True)
        # 파스텔톤 히트맵 색상 강조
        style_data_conditional = []
        for stat in stat_cols:
            vmin = table_df[stat].min()
            vmax = table_df[stat].max()
            for irow, row in table_df.iterrows():
                color = get_pastel_gradient_color(row[stat], vmin, vmax)
                style_data_conditional.append({
                    'if': {'row_index': irow, 'column_id': stat},
                    'backgroundColor': color,
                    'color': 'black'
                })
        output_components.extend([
            html.H4("Average Performance Change (%)", className="mt-3"),
            dash_table.DataTable(
                id={'type': 'change-table', 'index': i},
                data=table_df.round(4).to_dict('records'),
                columns=[{"name": c, "id": c} for c in table_df.columns],
                style_table={'overflowX': 'auto'},
                style_data_conditional=style_data_conditional,
                style_cell={'textAlign': 'center', 'minWidth': '80px', 'maxWidth': '100px', 'width': '90px'},
                row_selectable='single',
                cell_selectable=True,
                export_format='csv',
                export_headers='display',
            ),
            html.Div(id={'type': 'run-level-graph', 'index': i})
        ])
        # --- 그래프 추가 ---
        figures = []
        for stat in stat_cols:
            fig = px.bar(table_df, x='Subgroup', y=stat, title=f"{stat} Change (%)", labels={stat: 'Change (%)', 'Subgroup': 'Subgroup'})
            fig.update_layout(title_x=0.5)
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
    prevent_initial_call=True,
)
def show_run_level_table(active_cell, table_data, run_change_dfs_json, dir_names):
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
    # 파스텔톤 히트맵 적용
    stat_cols = [col for col in pivot_df.columns if col != 'Run']
    style_data_conditional = []
    for stat in stat_cols:
        vmin = pivot_df[stat].min()
        vmax = pivot_df[stat].max()
        for irow, row in pivot_df.iterrows():
            color = get_pastel_gradient_color(row[stat], vmin, vmax)
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
        fig = px.bar(df, x='Subgroup', y=stat, title=f"{stat} Change (%)", labels={stat: 'Change (%)', 'Subgroup': 'Subgroup'})
        fig.update_layout(title_x=0.5)
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
            fig = px.bar(stat_df, x='Subgroup', y='Value', title=f"{group} - {stat}", labels={'Value': stat, 'Subgroup': 'Subgroup'})
            fig.update_layout(title_x=0.5)
            figures.append(dbc.Col(dcc.Graph(figure=fig), width=12, lg=6, xl=4))
    return dbc.Row(figures)

# 분석 시작/완료 정보 표시 콜백 영어화
@app.callback(
    Output('analysis-info', 'children'),
    [Input('analysis-meta-store', 'data'),
     Input('summary-data-store', 'data')],
    State('job-start-time-store', 'data'),
    prevent_initial_call=True,
)
def update_analysis_info(meta, summary_data, start_time):
    if not meta:
        return ""
    dirs = meta.get('dirs', [])
    start_str = meta.get('start_str', '')
    info = [
        html.P(f"Target Directories: {', '.join(dirs) if dirs else 'None'}"),
        html.P(f"Analysis Start Time: {start_str}")
    ]
    if summary_data:
        elapsed = None
        if start_time:
            import time
            elapsed = time.time() - start_time
        msg = [html.P("Analysis completed!", style={"color": "#28a745", "fontWeight": "bold"})]
        if elapsed is not None:
            mins, secs = divmod(int(elapsed), 60)
            msg.append(html.P(f"Elapsed Time: {mins} min {secs} sec"))
        return info + msg
    else:
        return info + [html.P("Analysis in progress...", style={"color": "#007bff"})]

# 분석 시작 시 분석 대상/시작시간을 analysis-meta-store에 저장
@app.callback(
    Output('analysis-meta-store', 'data'),
    Input('run-button', 'n_clicks'),
    State('baseline-dropdown', 'value'),
    State('target-checklist', 'value'),
    State('manual-path-input', 'value'),
    State('job-start-time-store', 'data'),
    prevent_initial_call=True,
)
def store_analysis_meta(n_clicks, baseline_dir, target_dirs_checked, manual_paths, start_time):
    import datetime
    dirs = [baseline_dir] if baseline_dir else []
    if target_dirs_checked:
        dirs += list(target_dirs_checked)
    if manual_paths:
        dirs += [p.strip() for p in manual_paths.split('\n') if p.strip()]
    dirs = list(dict.fromkeys(dirs))
    if start_time:
        try:
            start_dt = datetime.datetime.fromtimestamp(start_time)
            start_str = start_dt.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            start_str = str(start_time)
    else:
        start_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    return {'dirs': dirs, 'start_str': start_str}


if __name__ == "__main__":
    app.run(debug=False)
