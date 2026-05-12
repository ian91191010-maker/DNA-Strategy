import os
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime

# 引入你已經寫好的 FinMind 資料抓取與指標引擎
from data_engine import fetch_finmind_data, process_all_indicators

class BigBullAuditEngine:
    def __init__(self):
        self.market_checks = {}
        self.pendulum_checks = {}

    # ---------------------------------------------------------
    # [1] 大盤環境與系統風險 (Mod A & G) - 完整還原原版邏輯
    # ---------------------------------------------------------
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
        
        # 月線轉折 N
        df_m = df.resample('ME').agg({'open':'first','close':'last','high':'max','low':'min'})
        df_m.index = df_m.index.to_period('M')
        current_period = df.index[-1].to_period('M')
        prev_1_period, prev_2_period = current_period - 1, current_period - 2
        
        n_high = max(df_m.loc[prev_1_period, 'high'], df_m.loc[prev_2_period, 'high'])
        n_low  = min(df_m.loc[prev_1_period, 'low'], df_m.loc[prev_2_period, 'low'])
        n_val  = (n_high + n_low) / 2
        n_low_val = (df_m.loc[prev_1_period, 'high'] + df_m.loc[prev_1_period, 'low']) / 2
        
        current_month_min = df[df.index.to_period('M') == current_period]['low'].min()
        is_extreme_bear = (current_month_min - n_val) < -600
        active_n = n_low_val if is_extreme_bear else n_val
        
        day_of_month = df.index[-1].day
        chk_timing = (5 <= day_of_month <= 10) if is_extreme_bear else (day_of_month >= 19)

        chk_05 = all([row[col_dif] > prev[col_dif], row[col_sig] > prev[col_sig], row['EMA200'] > prev['EMA200'], row['EMA209'] > prev['EMA209']])
        
        # 系統風險 Mod G
        lookback = 750 if len(df) >= 750 else len(df)
        wave_low = df['low'].tail(lookback).min()
        wave_high = df['high'].tail(lookback).max()
        chk_30_wave = (wave_high - wave_low) > 6000
        
        hist_dif_max = df[col_dif].rolling(lookback).max().iloc[-1]
        hist_dif_min = df[col_dif].rolling(lookback).min().iloc[-1]
        chk_33 = row[col_dif] >= (hist_dif_max * 0.95)
        chk_34_day_bottom = row[col_dif] <= (hist_dif_min * 0.95)
        
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

        mod_g_msgs = []
        if chk_33: mod_g_msgs.append("🔴 [高風險]")
        if chk_34_day_bottom: mod_g_msgs.append("🟢 [進場]")
        
        if bull_streak >= 6.0 and len(df_m) >= 2:
            curr_k, prev_k = df_m.iloc[-1], df_m.iloc[-2]
            A = curr_k['high'] - prev_k['high']
            B = curr_k['low'] - prev_k['low']
            
            if curr_k['high'] > prev_k['high']:
                resolution_x = prev_k['low'] - A
                if A < B: mod_g_msgs.append(f"月{bull_streak}K (A<B 化解)")
                elif curr_k['close'] < resolution_x: mod_g_msgs.append(f"月{bull_streak}K (破X點化解)")
                else: mod_g_msgs.append("⚠️ [誘多陷阱]")
            else:
                resolution_x = prev_k['low']
                if curr_k['close'] < resolution_x: mod_g_msgs.append(f"月{bull_streak}K (破X點化解)")
                else: mod_g_msgs.append("⚠️ [誘多陷阱]")

        elif bear_streak >= 6.0:
            mod_g_msgs.append(f"🚀 月{bear_streak}K 買點/轉機")

        is_safe = chk_timing and (row['close'] > active_n) and chk_05

        mod_g_str = "<br>".join(mod_g_msgs) if mod_g_msgs else "🟡 [常規操作]"

        self.market_checks = {
            "TSE_Close": round(row['close'], 2),
            "Active_N": round(active_n, 2),
            "Normal_N": round(n_val, 2),         # 新增：正常轉折點位
            "Low_N": round(n_low_val, 2),        # 新增：高低轉折點位
            "Is_Safe": is_safe,
            "Env_Light": "🟢 允許買進 (大盤安全)" if is_safe else "🔴 觀望/風控 (大盤修正)",
            "Streak_Msg": f"多頭 {bull_streak}K | 空頭 {bear_streak}K",
            "Chk_30_Wave": chk_30_wave,
            "Mod_G": mod_g_str                   # 套用新版換行字串
        }
        return self.market_checks

    # ---------------------------------------------------------
    # [2] 資金鐘擺效應 (Mod B) - 解決 yfinance 瓶頸，改用 FinMind
    # ---------------------------------------------------------
    def audit_pendulum(self, tse_close: float, finance_index: float) -> dict:
        import streamlit as st
        # 建立內部函式利用快取抓取 ETF 代理，避免每次執行都卡頓
        @st.cache_data(ttl=3600)
        def get_proxy_data(stock_id):
            return fetch_finmind_data(stock_id, years=3.0)

        proxies = {"權值(0050)": "0050", "中小(0051)": "0051", "金融(0055)": "0055"}
        sector_scores = {}
        winner_name = "N/A"
        z_val = 0
        mod_b_res = "Neutral"
        doomsday_status = "🟢 資金配置正常"
        
        # 1. 歷史動能掃描
        for name, code in proxies.items():
            df_p = get_proxy_data(code)
            if not df_p.empty:
                df_pm = df_p.resample('ME').agg({'high':'max','low':'min','close':'last'})
                if len(df_pm) >= 3:
                    wr3 = ta.willr(df_pm['high'], df_pm['low'], df_pm['close'], length=3)
                    if wr3 is not None and not wr3.empty:
                        sector_scores[name] = abs(wr3.iloc[-1]) # 取絕對值方便比較
        
        if sector_scores:
            winner_name = min(sector_scores, key=sector_scores.get)

        # 2. 小鐘擺 Z 值 (大盤 - 台積電*30)
        tsmc = get_proxy_data("2330")
        if not tsmc.empty and tse_close > 0:
            tsmc_close = tsmc['close'].iloc[-1]
            z_val = tse_close - (tsmc_close * 30)
            if z_val > 1000: mod_b_res = "Large (偏重權值)"
            elif z_val < -1000: mod_b_res = "Small (偏重中小)"

        # 3. 末日鐘擺
        doomsday_val = 0
        if self.market_checks.get("Chk_30_Wave", False) and finance_index > 0:
            doomsday_val = tse_close - (finance_index * 10)
            if doomsday_val > 2000: doomsday_status = "⚠️ 權值末日 (清倉權/中，轉入金融)"
            elif doomsday_val < -2000: doomsday_status = "⚠️ 金融末日 (轉入權值/中小)"
        elif not self.market_checks.get("Chk_30_Wave", False):
            doomsday_status = "未觸發 (波段未達6000點)"

        self.pendulum_checks = {
            "Doomsday_Val": round(doomsday_val, 2),
            "Doomsday_Status": doomsday_status,
            "Winner_Proxy": winner_name,
            "Z_Value": round(z_val, 2),
            "Mod_B_Status": mod_b_res
        }
        return self.pendulum_checks

    # ---------------------------------------------------------
    # [3] 類股板塊審計 (Mainstream Sector Analysis)
    # ---------------------------------------------------------
    def audit_mainstream_sectors(self, csv_path: str = 'twse_sector_indices.csv') -> list:
        """
        讀取 twse_quant_project.py 產出的 CSV，計算動能篩選出前五大主流類股
        """
        top_sectors = []
        try:
            if not os.path.exists(csv_path):
                return ["尚未產生類股指數 CSV", "請先執行 twse_quant_project.py"]

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
                
                # [條件一] 四箭頭向上
                cond1 = (dif.iloc[-1] > dif.iloc[-2]) and \
                        (hist.iloc[-1] > hist.iloc[-2]) and \
                        (ema200.iloc[-1] > ema200.iloc[-2]) and \
                        (ema209.iloc[-1] > ema209.iloc[-2])
                if not cond1: continue
                
                # [條件二] 月線 RSI(4) 排序
                series_m = series.resample('ME').last().dropna()
                if len(series_m) < 4: continue
                
                rsi4_m = ta.rsi(series_m, length=4)
                consecutive_months = 0
                for val in reversed(rsi4_m.values):
                    if pd.notna(val) and val >= 77: 
                        consecutive_months += 1
                    else: 
                        break
                
                candidates.append({
                    'Sector': sector_name,
                    'Consecutive': consecutive_months,
                    'RSI4': rsi4_m.iloc[-1] if pd.notna(rsi4_m.iloc[-1]) else 0
                })
                
            if candidates:
                df_cand = pd.DataFrame(candidates).sort_values(by=['Consecutive', 'RSI4'], ascending=[False, False])
                for rank, row in enumerate(df_cand.head(5).itertuples(), start=1):
                    top_sectors.append(f"{rank}. {row.Sector} (連{row.Consecutive}月, RSI:{row.RSI4:.1f})")
            else:
                top_sectors = ["目前無觸發四箭頭之強勢類股"]

        except Exception as e:
            top_sectors = [f"類股運算錯誤: {e}"]
            
        return top_sectors

    # ---------------------------------------------------------
    # [4] 個股 DNA 審計 (嚴格 Mod E 3+6 判定)
    # ---------------------------------------------------------
    def audit_stock_full(self, stock_id: str, drive_data: dict) -> dict:
        try:
            df = fetch_finmind_data(stock_id, years=15.0)
            if df.empty: return self._empty_result(stock_id, drive_data, "無K線資料")

            df_final = process_all_indicators(df)
            if df_final.empty: return self._empty_result(stock_id, drive_data, "指標運算失敗")
            
            # ADX_300 計算
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
            
            # --- [刪除 Mod C，合併為 Mod E 的 3 項前置條件] ---
            # C11: MACD 螺旋 (加入 MACD_HIST 與 SIG)
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
            
            # --- [六大跡象評分 (Mod E)] ---
            s1 = 1 if row.get('PLUS_DI_M_1', 0) > 50 else 0
            s2 = 1 if row.get('RSI_M_4', 0) > 77 else 0
            s3 = 1 if c19 else 0
            s4 = 1 if row.get('RSI_60', 0) > 57 else 0
            s5 = 1 if row.get('VR_W_2', 0) >= 150 else 0
            s6 = 1 if row.get('VR_M_2', 0) >= 150 else 0
            
            total_score = s1 + s2 + s3 + s4 + s5 + s6
            score_pass = total_score >= 3  # 嚴格大於等於 3
            
            # 最終 DNA 判定
            is_dna_pass = c11 and c12 and c19 and score_pass

            # --- [修正區塊：Mod D 切入訊號完整還原] ---
            d_signals = []
            tse_close = self.market_checks.get('TSE_Close', 0)
            active_n = self.market_checks.get('Active_N', 99999)
            
            # Check 13: 空頭乖離
            if row.get('RSI_60', 100) < 34: 
                d_signals.append("Chk13(空頭乖離)")
                
            # Check 14: 大盤位階判斷 (常規備戰區 vs 錯過低點)
            if (active_n - 100) <= tse_close <= active_n:
                d_signals.append("Chk14(常規進場/備戰區)")
            elif tse_close > active_n:
                d_signals.append("Chk14(錯過低點/需等突破)")
                
            # Check 15: 配合大盤反轉訊號
            mod_g_status = self.market_checks.get('Mod_G', '')
            if '日頂地' in mod_g_status or '空頭轉機' in mod_g_status:
                d_signals.append("Chk15(大盤反轉配合)")

            # --- [修正區塊：Mod F 加回系統風險與末日鐘擺連動] ---
            f_signals = []
            
            # 1. 個股技術面停損點
            if row.get('RSI_M_4', 100) < 77: f_signals.append("減碼50% (月RSI4<77)")
            if abs(row.get('WILLR_M_3', 0)) > 50: f_signals.append("全數賣出 (W%R3 跌破50)")
            
            # 2. 系統風險強制連動
            doomsday_status = self.pendulum_checks.get('Doomsday_Status', '')
            if '末日' in doomsday_status:
                f_signals.append(f"系統警戒 ({doomsday_status})")
                
            mod_g_status = self.market_checks.get('Mod_G', '')
            if '急漲危機' in mod_g_status or '紅色警戒' in mod_g_status:
                f_signals.append("大盤高檔警戒")

            old_p = float(drive_data.get('最舊收盤價', row['close'])) # 強制轉為浮點數
            true_change_pct = ((row['close'] - old_p) / old_p * 100) if old_p != 0 else 0.0

            return {
                "股號": stock_id,
                "股名": drive_data.get('股名', '未知'),
                "收盤價(最新日期)": round(row['close'], 2), # 絕對最新的市場現價
                "漲跌幅 (%)": round(true_change_pct, 2),    # 絕對真實的累積漲跌幅
                "判定狀態": "🟢 Pass" if is_dna_pass else "⚪ Fail",
                "跡象評分": total_score,
                "切入訊號": ", ".join(d_signals) if d_signals else "-",
                "風控賣出": " | ".join(f_signals) if f_signals else "續抱",
                "Check_Log": f"C11:{c11} C12:{c12} C19:{c19} (得分:{total_score})"
            }

        except Exception as e:
            return self._empty_result(stock_id, drive_data, f"運算錯誤: {str(e)}")

    def _empty_result(self, stock_id: str, drive_data: dict, reason: str) -> dict:
        """輔助函式：回傳空值結構"""
        return {
            "股號": stock_id,
            "股名": drive_data.get('股名', '未知'),
            "收盤價(最新日期)": drive_data.get('收盤價(最新日期)', 0.0),
            "漲跌幅 (%)": drive_data.get('漲跌幅 (%)', 0.0),
            "判定狀態": "🔴 Error",
            "跡象評分": 0,
            "切入訊號": "-",
            "風控賣出": "-",
            "Check_Log": reason
        }
