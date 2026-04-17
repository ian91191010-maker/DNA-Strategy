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
    # [1] 大盤環境與系統風險 (Mod A & G)
    # ---------------------------------------------------------
    def audit_market(self, df_tse: pd.DataFrame) -> dict:
        """
        執行大盤環境審計：包含 6K 統計、月線轉折 N、與系統風險 (Mod G)
        """
        if df_tse.empty or len(df_tse) < 300:
            return {"Is_Safe": False, "Msg": "大盤數據不足"}

        df = df_tse.copy()
        
        # 技術指標計算
        macd = df.ta.macd(fast=200, slow=209, signal=210)
        df = pd.concat([df, macd], axis=1)
        col_dif = 'MACD_200_209_210'
        df['EMA200'] = df.ta.ema(length=200)
        df['EMA209'] = df.ta.ema(length=209)
        
        row, prev = df.iloc[-1], df.iloc[-2]

        # 1. 月線轉折 N 邏輯
        df_m = df.resample('ME').agg({'open':'first','close':'last','high':'max','low':'min'})
        df_m.index = df_m.index.to_period('M')
        
        active_n = 0
        if len(df_m) >= 2:
            prev_p = df_m.index[-2]
            n_val = (df_m.loc[prev_p, 'high'] + df_m.loc[prev_p, 'low']) / 2
            active_n = n_val 

        # 2. 系統風險 (Mod G): 頂天立地
        wave_high = df['high'].tail(750).max()
        wave_low = df['low'].tail(750).min()
        is_day_top = row[col_dif] >= (df[col_dif].tail(750).max() * 0.95) if len(df) >= 750 else False
        
        # 3. 月 6K 統計
        bull_streak = 0
        for i in range(1, len(df_m)):
            if df_m.iloc[i]['low'] >= df_m.iloc[i-1]['low']: 
                bull_streak += 1
            else: 
                bull_streak = 0
                
        is_safe = (row['close'] > active_n) and (row[col_dif] > prev[col_dif])
        
        mod_g_msg = []
        if is_day_top: mod_g_msg.append("紅色警戒 (DIF日頂天)")
        if bull_streak >= 6: mod_g_msg.append(f"月 {bull_streak}K 高檔")

        self.market_checks = {
            "TSE_Close": round(row['close'], 2),
            "Active_N": round(active_n, 2),
            "Is_Safe": is_safe,
            "Env_Light": "🟢 允許買進 (大盤安全)" if is_safe else "🔴 觀望/風控 (大盤修正)",
            "Streak_Msg": f"月 {bull_streak}K" if bull_streak > 0 else "趨勢修正",
            "Mod_G": " | ".join(mod_g_msg) if mod_g_msg else "系統安全"
        }
        return self.market_checks

    # ---------------------------------------------------------
    # [2] 資金鐘擺效應 (Mod B)
    # ---------------------------------------------------------
    def audit_pendulum(self, tse_close: float, finance_index: float) -> dict:
        """
        計算末日鐘擺 (X-Y*10)
        """
        if finance_index <= 0:
            return {"Doomsday_Val": "-", "Doomsday_Status": "未輸入金融指數"}

        doomsday_val = tse_close - (finance_index * 10)
        
        doomsday_status = "🟢 資金配置正常"
        if doomsday_val > 2000: doomsday_status = "⚠️ 權值末日 (轉入金融)"
        elif doomsday_val < -2000: doomsday_status = "⚠️ 金融末日 (轉入權值/中小)"

        self.pendulum_checks = {
            "Doomsday_Val": round(doomsday_val, 2),
            "Doomsday_Status": doomsday_status
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
                consecutive_months = sum(1 for val in reversed(rsi4_m.values) if pd.notna(val) and val >= 77)
                
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
    # [4] 個股 DNA 審計 (Mod C, D, E, F) - 完全對齊 DNAstock.py
    # ---------------------------------------------------------
    def audit_stock_full(self, stock_id: str, drive_data: dict) -> dict:
        """
        執行 Check 11-19 (DNA)、六大跡象評分、與 Mod D/F 訊號
        """
        try:
            # 1. 抓取資料 (15年供 ADX300 暖機)
            df = fetch_finmind_data(stock_id, years=15.0)
            if df.empty:
                return self._empty_result(stock_id, drive_data, "無K線資料")

            # 2. 運算常規指標 (引用你原本的 data_engine)
            df_final = process_all_indicators(df)
            if df_final.empty:
                return self._empty_result(stock_id, drive_data, "指標運算失敗")
            
            # 3. 補齊 DNAstock.py 裡特製的 Wilder 平滑法 ADX_300
            high = df_final['high']
            low = df_final['low']
            close = df_final['close']
            
            tr1 = high - low
            tr2 = (high - close.shift(1)).abs()
            tr3 = (low - close.shift(1)).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            up = high - high.shift(1)
            down = low.shift(1) - low
            plus_dm = np.where((up > down) & (up > 0), up, 0.0)
            minus_dm = np.where((down > up) & (down > 0), down, 0.0)
            
            period = 300
            alpha = 1 / period
            atr = tr.ewm(alpha=alpha, adjust=False).mean()
            plus_di = 100 * pd.Series(plus_dm, index=df_final.index).ewm(alpha=alpha, adjust=False).mean() / atr
            minus_di = 100 * pd.Series(minus_dm, index=df_final.index).ewm(alpha=alpha, adjust=False).mean() / atr
            
            dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
            df_final['ADX_300'] = dx.ewm(alpha=alpha, adjust=False).mean()

            # 4. 資料清理
            df_final = df_final[~df_final.index.duplicated(keep='last')]
            df_final.sort_index(ascending=True, inplace=True)
            df_final.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            if len(df_final) < 2:
                return self._empty_result(stock_id, drive_data, "有效K線不足")

            row = df_final.iloc[-1]
            prev = df_final.iloc[-2]
            
            # --- [核心判定 Check 11, 12, 19] ---
            c11 = (row.get('MACD_DIF_1', 0) > prev.get('MACD_DIF_1', 0)) and (row.get('EMA200', 0) > prev.get('EMA200', 0))
            c12 = row['ADX_300'] > prev['ADX_300']
            c19 = row.get('WILLR_50', -100) > -20 # -20 代表強勢區
            
            # --- [Mod E: 六大跡象評分] ---
            s1 = 1 if row.get('PLUS_DI_M_1', 0) > 50 else 0
            s2 = 1 if row.get('RSI_M_4', 0) > 77 else 0
            s3 = 1 if c19 else 0
            s4 = 1 if row.get('RSI_60', 0) > 57 else 0
            s5 = 1 if row.get('VR_W_2', 0) >= 150 else 0
            s6 = 1 if row.get('VR_M_2', 0) >= 150 else 0
            
            total_score = s1 + s2 + s3 + s4 + s5 + s6
            score_pass = total_score >= 3
            
            # 最終 DNA 判定
            is_dna_pass = c11 and c12 and c19 and score_pass

            # --- [Mod D: 切入訊號] ---
            d_signals = []
            if row.get('RSI_60', 100) < 34: d_signals.append("空頭乖離")
            if self.market_checks.get('Is_Safe'): d_signals.append("常規進場")

            # --- [Mod F: 風控賣出] ---
            f_signals = []
            if row.get('RSI_M_4', 100) < 77: f_signals.append("減碼50% (月RSI4<77)")
            if abs(row.get('WILLR_M_3', 0)) > 50: f_signals.append("全數賣出 (W%R3 跌破50)")

            return {
                "股號": stock_id,
                "名稱": drive_data.get('證券名稱', '未知'),
                "當日收盤價": round(row['close'], 2),
                "總漲幅 (%)": drive_data.get('總漲幅 (%)', 0),
                "判定狀態": "🟢 Pass" if is_dna_pass else "⚪ Fail",
                "跡象評分": total_score,
                "切入訊號": ", ".join(d_signals) if d_signals else "-",
                "風控賣出": " | ".join(f_signals) if f_signals else "續抱",
                "Check_Log": f"C11:{c11} C12:{c12} C19:{c19} (s1:{s1} s2:{s2} s3:{s3} s4:{s4} s5:{s5} s6:{s6})"
            }

        except Exception as e:
            return self._empty_result(stock_id, drive_data, f"運算錯誤: {str(e)}")

    def _empty_result(self, stock_id: str, drive_data: dict, reason: str) -> dict:
        """輔助函式：回傳空值結構"""
        return {
            "股號": stock_id,
            "名稱": drive_data.get('證券名稱', '未知'),
            "當日收盤價": 0,
            "總漲幅 (%)": drive_data.get('總漲幅 (%)', 0),
            "判定狀態": "🔴 Error",
            "跡象評分": 0,
            "切入訊號": "-",
            "風控賣出": "-",
            "Check_Log": reason
        }