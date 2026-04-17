import requests
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime
from dateutil.relativedelta import relativedelta
import streamlit as st

# ==========================================
# 1. 核心資料抓取模組 (FinMind API)
# ==========================================
def fetch_finmind_data(stock_id: str, years: float = 15.0) -> pd.DataFrame:
    """
    從 FinMind 獲取台股日線資料，並自動嘗試讀取 Streamlit Secrets 中的 Token
    """
    # 1. 自動抓取保險箱裡的 Token，如果沒有就留空（使用免費版額度）
    fm_token = ""
    try:
        if "FINMIND_TOKEN" in st.secrets:
            fm_token = st.secrets["FINMIND_TOKEN"]
    except Exception:
        pass # 在非 Streamlit 環境下執行時的容錯機制

    # 2. 組合 API 請求
    url = "https://api.finmindtrade.com/api/v4/data"
    start_date = (datetime.now() - relativedelta(years=int(years))).strftime('%Y-%m-%d')
    parameter = {
        "dataset": "TaiwanStockPrice",
        "data_id": stock_id,
        "start_date": start_date,
        "token": fm_token  
    }

    # 3. 發送請求並解析資料
    try:
        resp = requests.get(url, params=parameter, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        if data.get("msg") == "success" and len(data.get("data", [])) > 0:
            df = pd.DataFrame(data["data"])
            
            # 標準化時間與欄位名稱
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # FinMind 的欄位名稱有時是 max/min，將其轉為標準的 high/low
            rename_map = {
                'Trading_Volume': 'volume',
                'max': 'high',
                'min': 'low'
            }
            df.rename(columns=rename_map, inplace=True)
            
            # 將欄位名稱全部轉小寫以防萬一
            df.columns = [c.lower() for c in df.columns]
            
            # 確保數值型態正確
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    
            return df
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"⚠️ 獲取 {stock_id} FinMind 資料失敗: {e}")
        return pd.DataFrame()

# ==========================================
# 2. 指標運算模組 (支援跨週期 K 線)
# ==========================================
def process_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    計算系統所需的所有日/週/月線指標。
    包含：EMA200, EMA209, MACD_DIF, WILLR_50, RSI_60, VR, 以及月線 RSI4 等。
    """
    if df.empty or len(df) < 50:
        return pd.DataFrame()

    df = df.copy()

    # ----------------------------------
    # [1] 日線級別指標 (Daily)
    # ----------------------------------
    df['EMA200'] = ta.ema(df['close'], length=200)
    df['EMA209'] = ta.ema(df['close'], length=209)

    # MACD (200, 209, 210)
    macd = ta.macd(df['close'], fast=200, slow=209, signal=210)
    if macd is not None and not macd.empty:
        df['MACD_DIF_1'] = macd.iloc[:, 0]  # 取 DIF 值

    # 威廉指標 W%R (50) 與 RSI (60)
    df['WILLR_50'] = ta.willr(df['high'], df['low'], df['close'], length=50)
    df['RSI_60'] = ta.rsi(df['close'], length=60)

    # ----------------------------------
    # 輔助函式：計算 VR (容量比率)
    # ----------------------------------
    def calculate_vr(df_period, length=2):
        if len(df_period) < length: 
            return pd.Series(index=df_period.index, dtype=float)
        diff = df_period['close'].diff()
        up_vol = df_period['volume'].where(diff > 0, 0)
        down_vol = df_period['volume'].where(diff < 0, 0)
        even_vol = df_period['volume'].where(diff == 0, 0)
        
        up_sum = up_vol.rolling(length).sum()
        down_sum = down_vol.rolling(length).sum()
        even_sum = even_vol.rolling(length).sum()
        
        # VR = (上漲量 + 0.5平盤量) / (下跌量 + 0.5平盤量) * 100
        vr = (up_sum + 0.5 * even_sum) / (down_sum + 0.5 * even_sum + 1e-9) * 100
        return vr

    # ----------------------------------
    # [2] 週線級別指標 (Weekly)
    # ----------------------------------
    # 轉換為週線 (以週五為基準)
    df_w = df.resample('W-FRI').agg({
        'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'
    }).dropna()
    
    df_w['VR_W_2'] = calculate_vr(df_w, length=2)

    # ----------------------------------
    # [3] 月線級別指標 (Monthly)
    # ----------------------------------
    # 轉換為月線 (以月底為基準)
    df_m = df.resample('ME').agg({
        'open':'first', 'high':'max', 'low':'min', 'close':'last', 'volume':'sum'
    }).dropna()
    
    df_m['VR_M_2'] = calculate_vr(df_m, length=2)
    df_m['RSI_M_4'] = ta.rsi(df_m['close'], length=4)
    df_m['WILLR_M_3'] = ta.willr(df_m['high'], df_m['low'], df_m['close'], length=3)

    # 月線 +DI (使用 14 期趨勢作為基準替代)
    adx_m = ta.adx(df_m['high'], df_m['low'], df_m['close'], length=14)
    if adx_m is not None and not adx_m.empty:
        dmp_col = [c for c in adx_m.columns if 'DMP' in c] # 尋找 +DI 欄位
        if dmp_col:
            df_m['PLUS_DI_M_1'] = adx_m[dmp_col[0]]

    # ----------------------------------
    # [4] 跨週期訊號對齊 (Align to Daily)
    # ----------------------------------
    # 將週線與月線算出來的數值，透過 forward-fill (向前填充) 補到每天的 K 線上
    df_w_aligned = df_w[['VR_W_2']].reindex(df.index, method='ffill')
    df_m_aligned = df_m[['VR_M_2', 'RSI_M_4', 'WILLR_M_3', 'PLUS_DI_M_1']].reindex(df.index, method='ffill')

    # 合併所有指標
    df = pd.concat([df, df_w_aligned, df_m_aligned], axis=1)

    # 清除無限大數值以防 JSON 崩潰
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

# ==========================================
# 3. 股票名稱輔助模組
# ==========================================
@st.cache_data(ttl=86400) # 快取 24 小時，避免重複查詢
def get_stock_name(stock_id: str) -> str:
    """
    獲取股票名稱 (用於圖表顯示)。
    為了避免拖慢速度，若只是大盤或查無資料，則回傳空字串。
    """
    if stock_id == "TAIEX":
        return "加權指數"
        
    try:
        url = "https://api.finmindtrade.com/api/v4/data"
        parameter = {
            "dataset": "TaiwanStockInfo",
            "data_id": stock_id
        }
        resp = requests.get(url, params=parameter, timeout=5)
        data = resp.json()
        if data.get("msg") == "success" and len(data.get("data", [])) > 0:
            return data["data"][0].get("stock_name", "")
    except Exception:
        pass
        
    return ""