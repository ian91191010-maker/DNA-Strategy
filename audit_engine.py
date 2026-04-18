import os
import time
import random
import requests
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, timedelta
import streamlit as st

# 引入資料引擎
from data_engine import fetch_finmind_data, process_all_indicators

class BigBullAuditEngine:
    def __init__(self):
        self.market_checks = {}
        self.pendulum_checks = {}
        self.stock_sector_map = {}  
        self.top_sectors_dict = {}  
        
        # 啟動時自動載入清單
        self._load_stock_mappings()

    def _load_stock_mappings(self):
        """讀取底層 CSV 清單，支援多種檔名容錯辨識"""
        # 涵蓋你上傳的原始檔名與簡化檔名
        files_to_check = [
            '上市證券清單.csv', 
            '上櫃證券清單.csv',
            '證券清單.xlsx - 上市證券.csv',
            '證券清單.xlsx - 上櫃證券.csv'
        ]
        
        for filename in files_to_check:
            if os.path.exists(filename):
                try:
                    df = pd.read_csv(filename, dtype=str, encoding='utf-8-sig')
                    code_col = next((c for c in df.columns if '代號' in c), None)
                    sector_col = next((c for c in df.columns if '產業' in c or '類別' in c), None)
                    
                    if code_col and sector_col:
                        mapping = dict(zip(df[code_col], df[sector_col]))
                        self.stock_sector_map.update(mapping)
                except Exception as e:
                    print(f"⚠️ 載入 {filename} 映射失敗: {e}")

    # =========================================================
    # [資料管線] TWSE 類股指數自動更新
    # =========================================================
    def _fetch_twse_sector_index(self, date_str: str, max_retries: int = 3) -> pd.DataFrame:
        url = "https://www.twse.com.tw/exchangeReport/MI_INDEX"
        params = {"response": "json", "date": date_str, "type": "IND"}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json"
        }
        for attempt in range(max_retries + 1):
            try:
                response = requests.get(url, params=params, headers=headers, timeout=15)
                data = response.json()
                if data.get("stat") != "OK": return pd.DataFrame()
                
                target_df = pd.DataFrame()
                if "tables" in data:
                    for table in data["tables"]:
                        fields = [str(f).strip() for f in table.get("fields", [])]
                        idx_c = [f for f in fields if "指數" in f and "收盤" not in f and "報酬" not in f]
                        close_c = [f for f in fields if "收盤" in f]
                        if idx_c and close_c:
                            temp = pd.DataFrame(table["data"], columns=fields)
                            target_df = temp[[idx_c[0], close_c[0]]].copy()
                            target_df.rename(columns={idx_c[0]: "Sector_Name", close_c[0]: "Close"}, inplace=True)
                            break
                if not target_df.empty:
                    target_df["Close"] = target_df["Close"].astype(str).str.replace(",", "").astype(float)
                    target_df.insert(0, "Date", date_str)
                return target_df
            except:
                time.sleep(5)
        return pd.DataFrame()

    def _maintain_sector_pipeline(self, csv_path: str, target_days: int = 210):
        tw_time = datetime.utcnow() + timedelta(hours=8)
        target_date = tw_time.replace(hour=0, minute=0, second=0, microsecond=0)
        new_data = []
        df_existing = pd.DataFrame()

        if os.path.exists(csv_path):
            df_existing = pd.read_csv(csv_path, dtype={"Date": str})
            latest_date_str = df_existing["Date"].max()
            latest_date = datetime.strptime(latest_date_str, "%Y%m%d")
        else:
            latest_date = None

        collected = 0
        while collected < target_days:
            if latest_date and target_date <= latest_date: break
            if target_date.weekday() < 5:
                df = self._fetch_twse_sector_index(target_date.strftime("%Y%m%d"))
                if not df.empty:
                    new_data.append(df)
                    collected += 1
                time.sleep(random.uniform(2, 4))
            target_date -= timedelta(days=1)

        if new_data:
            df_new = pd.concat(new_data, ignore_index=True)
            df_final = pd.concat([df_new, df_existing], ignore_index=True).drop_duplicates(subset=["Date", "Sector_Name"])
            df_final.sort_values(by=["Date", "Sector_Name"], ascending=[False, True], inplace=True)
            df_final.head(target_days * 35).to_csv(csv_path, index=False, encoding="utf-8-sig")

    # =========================================================
    # [審計模組] Mod A, B, Mainstream & E
    # =========================================================
    def audit_market(self, df_tse: pd.DataFrame) -> dict:
        if df_tse.empty or len(df_tse) < 300:
            return {"Is_Safe": False, "Msg": "大盤數據不足"}

        df = df_tse.copy()
        macd = df.ta.macd(fast=200, slow=209, signal=210)
        df = pd.concat([df, macd], axis=1)
        col_dif, col_sig = 'MACD_200_209_210', 'MACDs_200_209_210'
        df['EMA200'] = df.ta.ema(length=200)
        df['EMA209'] = df.ta.ema(length=209)
        
        row, prev = df.iloc[-1], df.iloc[-2]
        df_m = df.resample('ME').agg({'open':'first','close':'last','high':'max','low':'min'})
        df_m.index = df_m.index.to_period('M')
        
        current_period = df.index[-1].to_period('M')
        p1, p2 = current_period - 1, current_period - 2
        
        n_high = max(df_m.loc[p1, 'high'], df_m.loc[p2, 'high'])
        n_low  = min(df_m.loc[p1, 'low'], df_m.loc[p2, 'low'])
        n_val  = (n_high + n_low) / 2
        n_low_val = (df_m.loc[p1, 'high'] + df_m.loc[p1, 'low']) / 2
        
        current_month_min = df[df.index.to_period('M') == current_period]['low'].min()
        is_extreme_bear = (current_month_min - n_val) < -600
        active_n = n_low_val if is_extreme_bear else n_val
        
        day_of_month = df.index[-1].day
        chk_timing = (5 <= day_of_month <= 10) if is_extreme_bear else (day_of_month >= 19)
        chk_05 = all([row[col_dif] > prev[col_dif], row[col_sig] > prev[col_sig], row['EMA200'] > prev['EMA200'], row['EMA209'] > prev['EMA209']])

        # 完整還原 6K 連續計數邏輯
        bull_streak, bear_streak = 0.0, 0.0
        for i in range(1, len(df_m)):
            curr_k, prev_k = df_m.iloc[i], df_m.iloc[i-1]
            k_len = curr_k['high'] - curr_k['low']
            step = 0.5 if (abs(curr_k['close'] - curr_k['open']) <= (k_len * 0.1) if k_len > 0 else True) else 1.0
            
            if curr_k['low'] < prev_k['low']: bull_streak = 0.0
            else:
                if (curr_k['close'] > curr_k['open']) and (curr_k['close'] > prev_k['high']) and ((curr_k['close'] - prev_k['close']) > 300 or bull_streak >= 5.5):
                    bull_streak += step
                    
            if curr_k['high'] > prev_k['high']: bear_streak = 0.0
            else:
                if (curr_k['close'] < curr_k['open']) and (curr_k['close'] < prev_k['low']) and ((prev_k['close'] - curr_k['close']) > 300 or bear_streak >= 5.5):
                    bear_streak += step

        lookback = 750 if len(df) >= 750 else len(df)
        chk_30_wave = (df['high'].tail(lookback).max() - df['low'].tail(lookback).min()) > 6000
        
        hist_dif_max = df[col_dif].rolling(lookback).max().iloc[-1]
        hist_dif_min = df[col_dif].rolling(lookback).min().iloc[-1]
        chk_33 = row[col_dif] >= (hist_dif_max * 0.95)
        chk_34_day_bottom = row[col_dif] <= (hist_dif_min * 0.95)

        mod_g_msgs = []
        if chk_33: mod_g_msgs.append("[紅色警戒] DIF日頂天")
        if chk_34_day_bottom: mod_g_msgs.append("[綠色買點] 日頂地")
        
        if bull_streak >= 6.0 and len(df_m) >= 2:
            curr_k, prev_k = df_m.iloc[-1], df_m.iloc[-2]
            A = curr_k['high'] - prev_k['high']
            B = curr_k['low'] - prev_k['low']
            if curr_k['high'] > prev_k['high']:
                resolution_x = prev_k['low'] - A
                if A < B: mod_g_msgs.append(f"月{bull_streak}K (A<B 化解)")
                elif curr_k['close'] < resolution_x: mod_g_msgs.append(f"月{bull_streak}K (破X點化解)")
                else: mod_g_msgs.append(f"急漲危機! 月{bull_streak}K未破關")
            else:
                resolution_x = prev_k['low']
                if curr_k['close'] < resolution_x: mod_g_msgs.append(f"月{bull_streak}K (破X點化解)")
                else: mod_g_msgs.append(f"急漲危機! 月{bull_streak}K未破關")
        elif bear_streak >= 6.0:
            mod_g_msgs.append(f"[空頭轉機] 月{bear_streak}K")

        is_safe = chk_timing and (row['close'] > active_n) and chk_05

        self.market_checks = {
            "TSE_Close": round(row['close'], 2),
            "Active_N": round(active_n, 2),
            "Is_Safe": is_safe,
            "Env_Light": "🟢 允許買進" if is_safe else "🔴 觀望/風控",
            "Streak_Msg": f"多頭 {bull_streak}K | 空頭 {bear_streak}K",
            "Chk_30_Wave": chk_30_wave,
            "Mod_G": " | ".join(mod_g_msgs) if mod_g_msgs else "系統安全"
        }
        return self.market_checks

    def audit_pendulum(self, tse_close: float, finance_index: float) -> dict:
        @st.cache_data(ttl=3600)
        def get_proxy(sid): return fetch_finmind_data(sid, years=3.0)

        proxies = {"權值(0050)": "0050", "中小(0051)": "0051", "金融(0055)": "0055"}
        sector_scores = {}
        for name, code in proxies.items():
            df_p = get_proxy(code)
            if not df_p.empty:
                df_pm = df_p.resample('ME').agg({'high':'max','low':'min','close':'last'})
                if len(df_pm) >= 3:
                    wr3 = ta.willr(df_pm['high'], df_pm['low'], df_pm['close'], length=3)
                    if wr3 is not None and not wr3.empty:
                        sector_scores[name] = abs(wr3.iloc[-1]) 
        
        winner_name = min(sector_scores, key=sector_scores.get) if sector_scores else "N/A"
        
        tsmc = get_proxy("2330")
        z_val = tse_close - (tsmc['close'].iloc[-1] * 30) if not tsmc.empty else 0
        
        doomsday_status = "🟢 資金配置正常"
        doomsday_val = 0
        if self.market_checks.get("Chk_30_Wave") and finance_index > 0:
            doomsday_val = tse_close - (finance_index * 10)
            if doomsday_val > 2000: doomsday_status = "⚠️ 權值末日 (清倉權/中，轉入金融)"
            elif doomsday_val < -2000: doomsday_status = "⚠️ 金融末日 (轉入權/中小)"
        elif not self.market_checks.get("Chk_30_Wave"):
            doomsday_status = "未觸發 (波段未達6000點)"

        self.pendulum_checks = {
            "Doomsday_Status": doomsday_status, 
            "Doomsday_Val": round(doomsday_val, 2),
            "Winner_Proxy": winner_name,
            "Z_Value": round(z_val, 2)
        }
        return self.pendulum_checks

    def audit_mainstream_sectors(self, csv_path: str = 'twse_sector_indices.csv') -> list:
        st.toast("🔄 正在檢查並更新類股指數資料...", icon="⏳")
        self._maintain_sector_pipeline(csv_path)

        if not os.path.exists(csv_path):
            return ["⚠️ 無法建立或讀取類股指數 CSV"]

        df_raw_idx = pd.read_csv(csv_path, dtype=str)
        if df_raw_idx.empty: return ["類股資料為空"]

        date_col, sector_col, close_col = df_raw_idx.columns[0], df_raw_idx.columns[1], df_raw_idx.columns[2]
        df_raw_idx[close_col] = pd.to_numeric(df_raw_idx[close_col].str.replace(',', ''), errors='coerce')
        df_raw_idx[date_col] = pd.to_datetime(df_raw_idx[date_col], format='%Y%m%d', errors='coerce')
        
        df_idx = df_raw_idx.pivot_table(index=date_col, columns=sector_col, values=close_col)
        df_idx.sort_index(inplace=True)
        
        candidates = []
        for sector_name in df_idx.columns:
            series = pd.to_numeric(df_idx[sector_name], errors='coerce').dropna()
            if len(series) < 210: continue
            
            macd = ta.macd(series, fast=200, slow=209, signal=210)
            ema200 = ta.ema(series, length=200)
            ema209 = ta.ema(series, length=209)
            if macd is None or ema200 is None or ema209 is None: continue
            
            dif, hist = macd.iloc[:, 0], macd.iloc[:, 1]
            cond1 = (dif.iloc[-1] > dif.iloc[-2]) and (hist.iloc[-1] > hist.iloc[-2]) and \
                    (ema200.iloc[-1] > ema200.iloc[-2]) and (ema209.iloc[-1] > ema209.iloc[-2])
            if not cond1: continue
            
            series_m = series.resample('ME').last().dropna()
            if len(series_m) < 4: continue
            
            rsi4_m = ta.rsi(series_m, length=4)
            consecutive_months = sum(1 for val in reversed(rsi4_m.values) if pd.notna(val) and val >= 77)
            
            candidates.append({
                'Sector': sector_name,
                'Consecutive': consecutive_months,
                'RSI4': rsi4_m.iloc[-1] if pd.notna(rsi4_m.iloc[-1]) else 0
            })
            
        top_sectors = []
        if candidates:
            df_cand = pd.DataFrame(candidates).sort_values(by=['Consecutive', 'RSI4'], ascending=[False, False])
            for rank, row in enumerate(df_cand.head(5).itertuples(), start=1):
                top_sectors.append(f"{rank}. {row.Sector} (連{row.Consecutive}月, RSI:{row.RSI4:.1f})")
                self.top_sectors_dict[row.Sector] = rank
        else:
            top_sectors = ["目前無觸發四箭頭之強勢類股"]
            
        return top_sectors

    def audit_stock_full(self, stock_id: str, drive_data: dict) -> dict:
        """完整回傳包含價格、漲跌幅、切入風控的 DNA 字典"""
        try:
            df = fetch_finmind_data(stock_id, years=15.0)
            if df.empty: return self._empty_result(stock_id, drive_data, "無K線資料")

            df_final = process_all_indicators(df)
            if df_final.empty: return self._empty_result(stock_id, drive_data, "指標運算失敗")
            
            high, low, close = df_final['high'], df_final['low'], df_final['close']
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            up, down = high - high.shift(1), low.shift(1) - low
            plus_dm = np.where((up > down) & (up > 0), up, 0.0)
            minus_dm = np.where((down > up) & (down > 0), down, 0.0)
            
            period, alpha = 300, 1 / 300
            atr = tr.ewm(alpha=alpha, adjust=False).mean()
            plus_di = 100 * pd.Series(plus_dm, index=df_final.index).ewm(alpha=alpha, adjust=False).mean() / atr
            minus_di = 100 * pd.Series(minus_dm, index=df_final.index).ewm(alpha=alpha, adjust=False).mean() / atr
            
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
            df_final['ADX_300'] = dx.ewm(alpha=alpha, adjust=False).mean()

            df_final = df_final[~df_final.index.duplicated(keep='last')].sort_index()
            df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
            if len(df_final) < 2: return self._empty_result(stock_id, drive_data, "有效K線不足")

            row, prev = df_final.iloc[-1], df_final.iloc[-2]
            
            macd = ta.macd(df_final['close'], fast=200, slow=209, signal=210)
            dif_1, hist_1, sig_1 = 0, 0, 0
            prev_dif_1, prev_hist_1 = 0, 0
            if macd is not None and len(macd) >= 2:
                dif_1, hist_1, sig_1 = macd.iloc[-1, 0], macd.iloc[-1, 1], macd.iloc[-1, 2]
                prev_dif_1, prev_hist_1 = macd.iloc[-2, 0], macd.iloc[-2, 1]

            c11 = (dif_1 > prev_dif_1) and (hist_1 > prev_hist_1) and \
                  (row.get('EMA200', 0) > prev.get('EMA200', 0)) and \
                  (row.get('EMA209', 0) > prev.get('EMA209', 0)) and \
                  (dif_1 > sig_1)

            c12 = row['ADX_300'] > prev['ADX_300']
            c19 = row.get('WILLR_50', -100) > -20 
            
            s1 = 1 if row.get('PLUS_DI_M_1', 0) > 50 else 0
            s2 = 1 if row.get('RSI_M_4', 0) > 77 else 0
            s3 = 1 if c19 else 0
            s4 = 1 if row.get('RSI_60', 0) > 57 else 0
            s5 = 1 if row.get('VR_W_2', 0) >= 150 else 0
            s6 = 1 if row.get('VR_M_2', 0) >= 150 else 0
            
            total_score = s1 + s2 + s3 + s4 + s5 + s6
            score_pass = total_score >= 3
            is_dna_pass = c11 and c12 and c19 and score_pass

            # 買點與賣點訊號 (Mod D & F)
            d_signals = []
            if row.get('RSI_60', 100) < 34: d_signals.append("空頭乖離")
            if self.market_checks.get('Is_Safe'): d_signals.append("常規進場")

            f_signals = []
            if row.get('RSI_M_4', 100) < 77: f_signals.append("減碼50% (月RSI4<77)")
            if abs(row.get('WILLR_M_3', 0)) > 50: f_signals.append("全數賣出 (W%R3 跌破50)")

            # 產業映射與主流判定
            sector = self.stock_sector_map.get(str(stock_id), "N/A")

            # 完整回傳給 DataFrame 的所有欄位 (恢復你的數值顯示)
            return {
                "股號": stock_id,
                "名稱": drive_data.get('證券名稱', '未知'),
                "產業別": sector,
                "當日收盤價": round(row['close'], 2),
                "總漲幅 (%)": drive_data.get('總漲幅 (%)', 0),
                "判定狀態": "🟢 Pass" if is_dna_pass else "⚪ Fail",
                "跡象評分": total_score,
                "切入訊號": ", ".join(d_signals) if d_signals else "-",
                "風控賣出": " | ".join(f_signals) if f_signals else "續抱",
                "Check_Log": f"C11:{bool(c11)} C12:{bool(c12)} C19:{bool(c19)} (得分:{total_score})"
            }

        except Exception as e:
            return self._empty_result(stock_id, drive_data, f"運算錯誤: {str(e)}")

    def _empty_result(self, stock_id: str, drive_data: dict, reason: str) -> dict:
        return {
            "股號": stock_id,
            "名稱": drive_data.get('證券名稱', '未知'),
            "產業別": "N/A",
            "當日收盤價": 0,
            "總漲幅 (%)": drive_data.get('總漲幅 (%)', 0),
            "判定狀態": "🔴 Error",
            "跡象評分": 0,
            "切入訊號": "-",
            "風控賣出": "-",
            "Check_Log": reason
        }
