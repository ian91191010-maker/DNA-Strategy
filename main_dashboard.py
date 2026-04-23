import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import json
import datetime
from dateutil.relativedelta import relativedelta

# 引入我們開發的三大引擎
from data_engine import fetch_finmind_data, process_all_indicators, get_stock_name
from drive_engine import DriveDataEngine
from audit_engine import BigBullAuditEngine

# ==========================================
# 0. 網頁全域與暫存狀態設定 (Session State)
# ==========================================
st.set_page_config(page_title="飆股篩選", layout="wide")

# 初始化 Session State，確保點擊表格時資料不會消失
if 'selected_ticker' not in st.session_state:
    st.session_state['selected_ticker'] = "TAIEX"
if 'audit_results' not in st.session_state:
    st.session_state['audit_results'] = pd.DataFrame()
if 'market_env' not in st.session_state:
    st.session_state['market_env'] = {}
if 'sectors' not in st.session_state:
    st.session_state['sectors'] = []
if 'pendulum' not in st.session_state:
    st.session_state['pendulum'] = {}

# 設定你的 Google Drive 參數 (請修改為你的真實 ID)
DRIVE_KEY_PATH = 'credentials.json'
DRIVE_FOLDER_ID = '1SQze_GC0pDWf07fHCHOQJoATbPnyg7yX'

# --- 新增：初始化 Google Drive 引擎 (用來存取總表與自選股) ---
drive_engine_instance = DriveDataEngine(DRIVE_KEY_PATH, DRIVE_FOLDER_ID)

# --- 新增：初始化自選股 Session State ---
if 'watchlist_df' not in st.session_state:
    st.session_state['watchlist_df'] = drive_engine_instance.load_watchlist()

# --- 新增：自選股輔助函式 ---
def add_to_watchlist(stock_id, stock_name):
    df = st.session_state['watchlist_df']
    # 確保比對時都是字串
    if str(stock_id) not in df['stock_id'].values:
        new_row = pd.DataFrame({'stock_id': [str(stock_id)], 'stock_name': [stock_name]})
        updated_df = pd.concat([df, new_row], ignore_index=True)
        st.session_state['watchlist_df'] = updated_df
        drive_engine_instance.save_watchlist(updated_df)
        st.toast(f"已將 {stock_name} 加入自選！")
    else:
        st.warning("這檔股票已經在您的自選清單中囉！")

def remove_from_watchlist(stock_id):
    df = st.session_state['watchlist_df']
    updated_df = df[df['stock_id'] != str(stock_id)]
    st.session_state['watchlist_df'] = updated_df
    drive_engine_instance.save_watchlist(updated_df)
    st.toast("已成功移除自選股！")

# 初始化審計引擎
audit_engine = BigBullAuditEngine()

@st.cache_data(ttl=12 * 3600)
def run_full_system_audit(_audit_engine, finance_index, key_path, folder_id):
    """
    將大盤環境、資金鐘擺、主流類股以及所有個股的 DNA 審計全部打包。
    12 小時內重複執行時，會直接從記憶體秒傳結果，跳過所有 API 抓取與運算。
    """
    import streamlit as st
    
    # 1. 抓取大盤資料並審計
    df_tse = fetch_finmind_data("TAIEX", years=5.0)
    market_env = _audit_engine.audit_market(df_tse)
    pendulum = _audit_engine.audit_pendulum(market_env.get('TSE_Close', 0), finance_index)
    sectors = _audit_engine.audit_mainstream_sectors()
    
    # 2. 獲取 Drive 名單
    st.toast("正在與 Google Drive 進行同步...", icon="🔄")
    temp_drive_engine = DriveDataEngine(key_path, folder_id)
    master_list = temp_drive_engine.get_eligible_dna_stocks()
    
    # 3. 掃描所有個股並執行 DNA 審計
    results = []
    total_stocks = len(master_list)
    
    if total_stocks > 0:
        # 🌟 建立進度條與狀態文字 🌟
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, row in master_list.iterrows():
            current = i + 1
            # 更新畫面上的進度與文字
            status_text.text(f"正在運算個股: {row['股號']} {row['股名']} ({current}/{total_stocks})")
            progress_bar.progress(current / total_stocks)
            
            # 執行審計
            res = _audit_engine.audit_stock_full(row['股號'], row.to_dict())
            results.append(res)
            
        # 跑完後清空進度條，讓畫面保持乾淨
        status_text.empty()
        progress_bar.empty()
            
    return market_env, pendulum, sectors, pd.DataFrame(results)

# ==========================================
# 1. 圖表渲染函式 (封裝 DNAstock.py 的邏輯)
# ==========================================
def render_interactive_chart(stock_id, years_to_show):
    try:
        stock_name = get_stock_name(stock_id)
        display_title = f"{stock_id} {stock_name}" if stock_name else stock_id
        
        # 抓取資料並運算
        df = fetch_finmind_data(stock_id, years=15.0) 
        df_final = process_all_indicators(df)
        
        if df_final.empty:
            st.error("該區間內沒有產生有效的數據！")
            return

        # 補算客製化 ADX_300 (維持你原本的邏輯)
        high, low, close = df_final['high'], df_final['low'], df_final['close']
        tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
        up, down = high - high.shift(1), low.shift(1) - low
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)
        alpha = 1 / 300
        atr = tr.ewm(alpha=alpha, adjust=False).mean()
        plus_di = 100 * pd.Series(plus_dm, index=df_final.index).ewm(alpha=alpha, adjust=False).mean() / atr
        minus_di = 100 * pd.Series(minus_dm, index=df_final.index).ewm(alpha=alpha, adjust=False).mean() / atr
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        df_final['ADX_300'] = dx.ewm(alpha=alpha, adjust=False).mean()

        # 資料清洗與時間裁切
        df_final = df_final[~df_final.index.duplicated(keep='last')].sort_index()
        cutoff_date = pd.to_datetime(datetime.date.today() - relativedelta(years=int(years_to_show), months=int((years_to_show % 1) * 12)))
        df_final = df_final[df_final.index >= cutoff_date].dropna(subset=['WILLR_50', 'PLUS_DI_M_1'])
        
        if df_final.empty:
            st.error("裁切後無有效數據，請拉長顯示區間。")
            return
            
        df_final['time'] = df_final.index.strftime('%Y-%m-%d')
        
        # 準備 JSON 資料
        wr_data, score_data, volume_dict = [], [], {}
        for _, row in df_final.iterrows():
            date_str = row['time']
            volume_dict[date_str] = int(row['volume'])
            wr_val = row['WILLR_50']
            
            s1 = 1 if row.get('PLUS_DI_M_1', 0) > 50 else 0
            s2 = 1 if row.get('RSI_M_4', 0) > 77 else 0
            s3 = 1 if wr_val > -20 else 0
            s4 = 1 if row.get('RSI_60', 0) > 57 else 0
            s5 = 1 if row.get('VR_W_2', 0) >= 150 else 0
            s6 = 1 if row.get('VR_M_2', 0) >= 150 else 0
            
            m4_score = s1 + s2 + s3 + s4 + s5 + s6
            wr_data.append({'time': date_str, 'value': wr_val})
            score_data.append({'time': date_str, 'value': m4_score, 'color': 'rgba(38, 166, 154, 0.8)' if m4_score >= 3 else 'rgba(239, 83, 80, 0.8)'})

        candles_json = json.dumps(df_final[['time', 'open', 'high', 'low', 'close']].to_dict(orient='records'))
        ema200_json = json.dumps(df_final[['time', 'EMA200']].rename(columns={'EMA200': 'value'}).to_dict(orient='records'))
        ema209_json = json.dumps(df_final[['time', 'EMA209']].rename(columns={'EMA209': 'value'}).to_dict(orient='records'))
        dif_json = json.dumps(df_final[['time', 'MACD_DIF_1']].rename(columns={'MACD_DIF_1': 'value'}).to_dict(orient='records'))
        adx_json = json.dumps(df_final[['time', 'ADX_300']].rename(columns={'ADX_300': 'value'}).to_dict(orient='records'))
        wr_json = json.dumps(wr_data)
        score_json = json.dumps(score_data)
        volume_json = json.dumps(volume_dict)

        # 組合 HTML 程式碼
        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <script src="https://unpkg.com/lightweight-charts@4.1.1/dist/lightweight-charts.standalone.production.js"></script>
            <style>
                body {{ margin: 0; padding: 0; background-color: #131722; color: white; font-family: sans-serif; overflow: hidden; }}
                #tvchart-container {{ position: relative; width: 100vw; height: 100vh; }}
                #tvchart {{ width: 100%; height: 100% }} 
                
                /* 透明灰色提示框 */
                #tooltip {{ 
                    position: absolute; z-index: 1000; background: rgba(50, 50, 50, 0.7); 
                    color: #E0E3EB; padding: 10px; border-radius: 6px; display: none; 
                    pointer-events: none; font-size: 13px; line-height: 1.5;
                    border: 1px solid rgba(255, 255, 255, 0.1); box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                }}
                
                #overlay-header {{
                    position: absolute;
                    top: 10px;
                    left: 15px;
                    right: 65px;
                    z-index: 10;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    pointer-events: none;
                }}
                
                #chart-title {{ 
                    color: #E0E3EB; 
                    font-size: 18px;
                    font-weight: bold; 
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    margin-right: 10px;
                }}
                
                #toolbar {{ 
                    display: flex; 
                    gap: 4px; 
                    pointer-events: auto;
                }}
                
                .tool-btn {{
                    background: rgba(43, 43, 67, 0.8); 
                    border: 1px solid #454559; 
                    color: #E0E3EB; 
                    cursor: pointer; 
                    padding: 4px 8px;  
                    border-radius: 4px; 
                    font-size: 13px;    
                    transition: background 0.2s;
                }}
                .tool-btn:hover {{ background: rgba(70, 70, 100, 1); }}
                .tool-btn.active {{
                    background: rgba(33, 150, 243, 0.8);
                    border-color: #2196F3;
                }}

                #selection-box {{
                    position: absolute;
                    border: 1px solid #2196F3;
                    background: rgba(33, 150, 243, 0.2);
                    display: none;
                    z-index: 100;
                    pointer-events: none;
                }}
                .tool-btn.active {{
                    background: rgba(33, 150, 243, 0.8);
                    border-color: #2196F3;
                }}
            </style>
        </head>
        <body>
            <div id="tvchart-container">
                <div id="overlay-header">
                    <div id="chart-title">{display_title}</div>
                    <div id="toolbar">
                        <button class="tool-btn" id="btn-zoom-box" title="區域放大">⬚</button>
                        <button class="tool-btn" id="btn-zoom-in" title="放大">＋</button>
                        <button class="tool-btn" id="btn-zoom-out" title="縮小">－</button>
                        <button class="tool-btn" id="btn-reset" title="重設視角">↺</button>
                        <button class="tool-btn" id="btn-fullscreen" title="全螢幕">⤢</button>
                    </div>
                </div>
                
                <div id="selection-box"></div>
                <div id="tooltip"></div>
                <div id="tvchart"></div>
            </div>
            <script>
                const container = document.getElementById('tvchart-container');
                const tooltip = document.getElementById('tooltip');
                const volumeData = {volume_json};

                const chart = LightweightCharts.createChart(document.getElementById('tvchart'), {{
                    layout: {{ background: {{ type: 'solid', color: '#131722' }}, textColor: '#d1d4dc' }},
                    grid: {{ vertLines: {{ color: '#2B2B43' }}, horzLines: {{ color: '#2B2B43' }} }},
                    crosshair: {{ mode: LightweightCharts.CrosshairMode.Normal }},
                    timeScale: {{ rightOffset: 80 }}
                }});

                // 主圖 K 線 (右側價格軸 1)
                chart.priceScale('right').applyOptions({{ scaleMargins: {{ top: 0.05, bottom: 0.52 }} }});
                const candleSeries = chart.addCandlestickSeries({{ upColor: '#ef5350', downColor: '#26a69a', borderVisible: false, wickUpColor: '#ef5350', wickDownColor: '#26a69a' }});
                candleSeries.setData({candles_json});

                // 均線 EMA
                chart.addLineSeries({{ color: '#f5c211', lineWidth: 2 }}).setData({ema200_json});
                chart.addLineSeries({{ color: '#e0591b', lineWidth: 2 }}).setData({ema209_json});

                // DIF 指標 (副圖 1)
                const difSeries = chart.addLineSeries({{ color: '#2962FF', lineWidth: 2, priceScaleId: 'dif_scale' }});
                chart.priceScale('dif_scale').applyOptions({{ scaleMargins: {{ top: 0.49, bottom: 0.39 }} }});
                difSeries.setData({dif_json});

                // ADX_300 指標 (副圖 2)
                const adxSeries = chart.addLineSeries({{ color: '#FF1493', lineWidth: 2, priceScaleId: 'adx_scale' }});
                chart.priceScale('adx_scale').applyOptions({{ scaleMargins: {{ top: 0.62, bottom: 0.26 }} }});
                adxSeries.setData({adx_json});

                // W%R 指標 (副圖 3)
                const wrSeries = chart.addLineSeries({{ color: '#00BCD4', lineWidth: 2, priceScaleId: 'wr_scale' }});
                chart.priceScale('wr_scale').applyOptions({{ scaleMargins: {{ top: 0.75, bottom: 0.13 }} }});
                wrSeries.setData({wr_json});
                wrSeries.createPriceLine({{ price: -20, color: '#FF9800', lineStyle: 2 }});

                // 跡象評分直方圖 (副圖 4)
                const scoreSeries = chart.addHistogramSeries({{ priceScaleId: 'score_scale' }});
                chart.priceScale('score_scale').applyOptions({{ scaleMargins: {{ top: 0.88, bottom: 0.0 }} }});
                scoreSeries.setData({score_json});
                scoreSeries.createPriceLine({{ price: 3, color: '#FFEB3B', lineStyle: 0 }});

                // 十字線互動邏輯
                chart.subscribeCrosshairMove((param) => {{
                    if (!param.time || !param.point || param.point.x < 0 || param.point.y < 0) {{
                        tooltip.style.display = 'none';
                        return;
                    }}
                    const data = param.seriesData.get(candleSeries);
                    if (!data) {{
                        tooltip.style.display = 'none';
                        return;
                    }}

                    // 1. 解決陷阱二：將圖表回傳的日期物件，轉回 YYYY-MM-DD 字串
                    let dateStr = param.time;
                    if (typeof param.time === 'object') {{
                        const y = param.time.year;
                        const m = String(param.time.month).padStart(2, '0');
                        const d = String(param.time.day).padStart(2, '0');
                        dateStr = y + '-' + m + '-' + d;
                    }}

                    tooltip.style.display = 'block';
                    
                    // 用正確的日期字串去字典抓取原始成交量 (股)
                    const rawVol = volumeData[dateStr] || 0; 
                    
                    // 2. 解決陷阱一：將「股」轉換為「張」
                    const volLots = rawVol / 1000; 

                    // 3. 處理 K 單位顯示邏輯 (大於等於 1000 張才顯示 K)
                    let volK = volLots.toString();
                    if (volLots >= 1000) {{
                        volK = (volLots / 1000).toFixed(1).replace(/\.0$/, '') + 'K';
                    }}

                    // 更新提示框內容 (注意這裡使用 dateStr)
                    tooltip.innerHTML = `
                        <div style="font-weight: bold; color: #f5c211; margin-bottom: 4px;">${{dateStr}}</div>
                        <div style="display: grid; grid-template-columns: auto auto; gap: 4px 12px;">
                            <span>開: <b style="color: #fff">${{data.open}}</b></span>
                            <span>高: <b style="color: #fff">${{data.high}}</b></span>
                            <span>低: <b style="color: #fff">${{data.low}}</b></span>
                            <span>收: <b style="color: #fff">${{data.close}}</b></span>
                        </div>
                        <div style="margin-top: 5px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 5px;">
                            成交量: <b style="color: #00BCD4">${{volK}}</b>
                        </div>
                    `;

                    // 智慧位置定位
                    const width = 180; const height = 100;
                    let left = param.point.x + 15; let top = param.point.y + 15;
                    if (left + width > container.clientWidth) left = param.point.x - width - 15;
                    if (top + height > container.clientHeight) top = param.point.y - height - 15;
                    tooltip.style.left = left + 'px'; tooltip.style.top = top + 'px';
                }});

                // 功能按鈕實作
                document.getElementById('btn-zoom-in').onclick = () => {{
                    const range = chart.timeScale().getVisibleLogicalRange();
                    if (range) {{
                        const diff = range.to - range.from;
                        chart.timeScale().setVisibleLogicalRange({{ from: range.from + diff * 0.15, to: range.to - diff * 0.15 }});
                    }}
                }};
                document.getElementById('btn-zoom-out').onclick = () => {{
                    const range = chart.timeScale().getVisibleLogicalRange();
                    if (range) {{
                        const diff = range.to - range.from;
                        chart.timeScale().setVisibleLogicalRange({{ from: range.from - diff * 0.15, to: range.to + diff * 0.15 }});
                    }}
                }};
                document.getElementById('btn-reset').onclick = () => chart.timeScale().fitContent();
                document.getElementById('btn-fullscreen').onclick = () => {{
                    if (!document.fullscreenElement) container.requestFullscreen();
                    else document.exitFullscreen();
                }};

                // RWD 尺寸調整
                new ResizeObserver(entries => {{
                    if (entries.length === 0) return;
                    const newRect = entries[0].contentRect;
                    chart.applyOptions({{ width: newRect.width, height: newRect.height }});
                }}).observe(container);

                // 區域放大 (Zoom Box) 核心邏輯
                const selectionBox = document.getElementById('selection-box');
                const btnZoomBox = document.getElementById('btn-zoom-box');
                let isZoomBoxMode = false;
                let isDrawing = false;
                let startX = 0;

                btnZoomBox.onclick = () => {{
                    isZoomBoxMode = !isZoomBoxMode;
                    btnZoomBox.classList.toggle('active', isZoomBoxMode);
                    container.style.cursor = isZoomBoxMode ? 'crosshair' : 'default';
                }};

                container.addEventListener('mousedown', (e) => {{
                    if (!isZoomBoxMode) return;
                    isDrawing = true;
                    const rect = container.getBoundingClientRect();
                    startX = e.clientX - rect.left;
                    
                    selectionBox.style.display = 'block';
                    selectionBox.style.left = startX + 'px';
                    selectionBox.style.width = '0px';
                    selectionBox.style.top = '0px';
                    selectionBox.style.height = '100%';
                }});

                container.addEventListener('mousemove', (e) => {{
                    if (!isDrawing) return;
                    const rect = container.getBoundingClientRect();
                    const currentX = e.clientX - rect.left;
                    
                    const left = Math.min(startX, currentX);
                    const width = Math.abs(currentX - startX);
                    
                    selectionBox.style.left = left + 'px';
                    selectionBox.style.width = width + 'px';
                }});

                window.addEventListener('mouseup', (e) => {{
                    if (!isDrawing) return;
                    isDrawing = false;
                    selectionBox.style.display = 'none';

                    const rect = container.getBoundingClientRect();
                    const endX = e.clientX - rect.left;

                    const timeScale = chart.timeScale();
                    const logicalStart = timeScale.coordinateToLogical(startX);
                    const logicalEnd = timeScale.coordinateToLogical(endX);

                    // 執行縮放 (確保有選取到超過1根K線)
                    if (logicalStart !== null && logicalEnd !== null && Math.abs(logicalEnd - logicalStart) > 1) {{
                        timeScale.setVisibleLogicalRange({{
                            from: Math.min(logicalStart, logicalEnd),
                            to: Math.max(logicalStart, logicalEnd)
                        }});
                    }}

                    // 自動關閉放大模式，恢復一般拖曳
                    isZoomBoxMode = false;
                    btnZoomBox.classList.remove('active');
                    container.style.cursor = 'default';
                }});

                chart.timeScale().fitContent();
            </script>
        </body>
        </html>
        """
        components.html(html_code, height=770, scrolling=False)
    except Exception as e:
        st.error(f"圖表繪製發生錯誤：{e}")

# ==========================================
# 2. 側邊欄 (Sidebar)
# ==========================================
with st.sidebar:
    st.markdown("<h2 style='text-align: center;'>飆股篩選系統</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    st.header("單檔探索")
    input_stock_id = st.text_input("輸入股票代碼", value="")
    years_to_show = st.slider("顯示資料區間 (年)", min_value=1.0, max_value=5.0, value=4.0, step=0.5)
    btn_single_search = st.button("顯示單檔圖表")
    
    # 邏輯：按下按鈕時，更新 Session State
    if btn_single_search and input_stock_id:
        st.session_state['selected_ticker'] = input_stock_id
    
    st.markdown("---")
    
    st.header("系統審計")
    finance_index_input = st.number_input("今日金融保險類指數 (Y)", min_value=0.0, value=0.0, step=0.1)
    btn_run_audit = st.button("啟動 / 更新雲端資料", type="primary", use_container_width=True)

    if st.button("清除快取強制重抓", use_container_width=True):
        st.cache_data.clear()
        st.success("✅ 快取已清除，請再次點擊上方按鈕更新資料！")

# ==========================================
# 3. 主畫面 (Main Content) 三層式版面配置
# ==========================================
st.markdown("<h3 style='text-align: center;'>大飆股 DNA 系統審計看板</h3>", unsafe_allow_html=True)

# ------------------------------------------
# 第一層：K 線圖區域
# ------------------------------------------
layer1_kline = st.container()
with layer1_kline:
    with st.container(border=True):
        st.subheader("K 線與指標")
        if st.session_state['selected_ticker']:
            with st.spinner(f"正在繪製 {st.session_state['selected_ticker']} 圖表..."):
                render_interactive_chart(st.session_state['selected_ticker'], years_to_show)
        else:
            st.markdown("<div style='height: 300px; display: flex; align-items: center; justify-content: center; color: gray;'>等待選擇標的...</div>", unsafe_allow_html=True)

# --- 將原本的按鈕更新邏輯放在第一層下方 ---
if btn_run_audit:
    if finance_index_input == 0.0:
        st.sidebar.warning("⚠️ 請記得輸入金融保險類指數！")
    else:
        with st.spinner("系統運算中，正在從雲端快取或 Drive 同步資料..."):
            try:
                # 直接呼叫帶有快取的巨型函式
                m_env, p_env, secs, df_res = run_full_system_audit(
                    audit_engine, 
                    finance_index_input, 
                    DRIVE_KEY_PATH, 
                    DRIVE_FOLDER_ID
                )
                
                st.session_state['market_env'] = m_env
                st.session_state['pendulum'] = p_env
                st.session_state['sectors'] = secs
                st.session_state['audit_results'] = df_res
                st.toast("審計完成！", icon="✅")
            except Exception as e:
                st.error(f"執行發生錯誤: {e}")

st.divider() # 加入分隔線

# ------------------------------------------
# 第二層：我的自選股區域
# ------------------------------------------
layer2_watchlist = st.container()
with layer2_watchlist:
    st.subheader("我的自選股")
    
    watch_df = st.session_state['watchlist_df']
    
    if watch_df.empty:
        st.info("目前清單空空的，快去下方挑選潛力股吧！")
    else:
        # 將畫面切為左右，左邊看表格，右邊做操作
        col_w1, col_w2 = st.columns([2, 1])
        with col_w1:
            st.dataframe(watch_df, use_container_width=True, hide_index=True)
            
        with col_w2:
            st.write("**自選股操作**")
            target_stock = st.selectbox(
                "請選擇要操作的自選股：", 
                options=watch_df['stock_id'].tolist(),
                format_func=lambda x: f"{x} - {watch_df[watch_df['stock_id']==x]['stock_name'].values[0]}"
            )
            
            c1, c2 = st.columns(2)
            with c1:
                if st.button("載入 K 線圖", key="btn_load_watch_kline"):
                    st.session_state['selected_ticker'] = target_stock
                    st.rerun() # 更新完標的後重新整理畫面畫圖
            with c2:
                if st.button("移除此檔", key="btn_remove_watch"):
                    remove_from_watchlist(target_stock)
                    st.rerun()

st.divider() # 加入分隔線

# ------------------------------------------
# 第三層：大盤環境與強勢股名單 (左右分欄)
# ------------------------------------------
layer3_bottom = st.container()
with layer3_bottom:
    col1, col2 = st.columns([1, 2])

    # 左欄：大盤環境 (保留你原本精美的 HTML 版面)
    with col1:
        st.subheader("大盤環境與風險")
        with st.container(border=True):
            m_env = st.session_state.get('market_env', {})
            p_env = st.session_state.get('pendulum', {})
            
            import re
            raw_streak = m_env.get('Streak_Msg', '-')
            streak_msg = re.sub(r'\d+\.\d+', lambda x: str(int(float(x.group()))), raw_streak)
            
            sectors = st.session_state.get('sectors', [])
            sectors_html = "".join([f"<li>{s}</li>" for s in sectors]) if sectors else "<li>尚未運算</li>"
            
            html_layout = f"""
            <div style="background-color: rgba(38, 166, 154, 0.5); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                <h4 style="margin: 0 0 10px 0; color: white;">環境燈號</h4>
                <span style="font-size: 16px; color: white;">{m_env.get('Env_Light', '尚未運算')}</span>
            </div>
            
            <div style="background-color: rgba(239, 83, 80, 0.5); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                <h4 style="margin: 0 0 10px 0; color: white;">轉折與趨勢</h4>
                <ul style="margin: 0; padding-left: 20px; color: white;">
                    <li>{streak_msg}</li>
                    <li>系統風險: {m_env.get('Mod_G', '-')}</li>
                </ul>
            </div>
            
            <div style="background-color: rgba(255, 152, 0, 0.5); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                <h4 style="margin: 0 0 10px 0; color: white;">末日鐘擺 (X-Y*10)</h4>
                <ul style="margin: 0; padding-left: 20px; color: white;">
                    <li>狀態: {p_env.get('Doomsday_Status', '-')}</li>
                    <li>差值: {p_env.get('Doomsday_Val', '-')}</li>
                </ul>
            </div>
            
            <div style="background-color: rgba(41, 98, 255, 0.5); padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                <h4 style="margin: 0 0 10px 0; color: white;">五大主流類股</h4>
                <ul style="margin: 0; padding-left: 20px; color: white;">
                    {sectors_html}
                </ul>
            </div>
            """
            st.markdown(html_layout, unsafe_allow_html=True)

    with col2:
        st.subheader("強勢股名單")
        with st.container(border=True):
            df_res = st.session_state['audit_results']
            
            # 檢查是否有資料
            if not df_res.empty:
                # 準備要顯示的表格資料
                df_res = df_res.sort_values(by='漲跌幅 (%)', ascending=False).reset_index(drop=True)
                display_df = df_res[['股號', '股名', '收盤價(最新日期)', '漲跌幅 (%)']]
                
                # 畫出可互動的表格，並將結果存入 event
                event = st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    selection_mode="single-row",
                    on_select="rerun"
                )
                
                # 捕捉點擊事件
                if len(event.selection.rows) > 0:
                    selected_idx = event.selection.rows[0]
                    clicked_ticker = str(display_df.iloc[selected_idx]["股號"]) 
                    clicked_name = str(display_df.iloc[selected_idx]["股名"])
                    
                    # 1. 自動更新上方 K 線圖 
                    if clicked_ticker != st.session_state.get('selected_ticker', ''):
                        st.session_state['selected_ticker'] = clicked_ticker
                        st.rerun() 
                    
                    # 2. 顯示按鈕
                    st.markdown(f"目前選中：**{clicked_name} ({clicked_ticker})**")
                    
                    # 點擊後將股票加入自選並強制重新整理畫面
                    if st.button("加入自選", key="btn_add_to_watch"):
                        add_to_watchlist(clicked_ticker, clicked_name)
                        st.rerun()
                        
            else:
                st.info("請點擊左側「啟動 / 更新雲端資料」載入本日名單。")
