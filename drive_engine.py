import io
import os
import re
import pandas as pd
import numpy as np
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

class DriveDataEngine:
    def __init__(self, credentials_path: str, folder_id: str):
        """
        初始化 Google Drive 連線引擎
        :param credentials_path: JSON 金鑰的路徑 (例如 'credentials.json')
        :param folder_id: Google Drive 資料夾的 ID
        """
        self.credentials_path = credentials_path
        self.folder_id = folder_id
        self.scopes = ['https://www.googleapis.com/auth/drive']
        self.service = self._authenticate()

    def _authenticate(self):
        """載入服務帳戶憑證並建立 API 服務物件 (支援本機與雲端環境)"""
        import streamlit as st
        from google.oauth2 import service_account
        
        # 模式 1：嘗試從 Streamlit Secrets 讀取 (雲端部署 / 本機 .streamlit 模式)
        try:
            if "gcp_service_account" in st.secrets:
                creds = service_account.Credentials.from_service_account_info(
                    st.secrets["gcp_service_account"], scopes=self.scopes
                )
                return build('drive', 'v3', credentials=creds)
        except Exception:
            pass # 如果讀不到 Secrets，就進入模式 2
            
        # 模式 2：退回使用本地端 JSON 檔案 (舊版備用模式)
        if os.path.exists(self.credentials_path):
            creds = service_account.Credentials.from_service_account_file(
                self.credentials_path, scopes=self.scopes)
            return build('drive', 'v3', credentials=creds)
        else:
            raise FileNotFoundError("⚠️ 找不到 API 憑證！請確認是否設定了 Streamlit Secrets 或 credentials.json。")

    # ==========================================
    # 檔案抓取與解析 (Excel)
    # ==========================================
    def fetch_dna_excel_files(self) -> list:
        """列出資料夾內所有符合 DNA成分股_YYYYMMDD.xlsx 的檔案，由新到舊排序"""
        query = f"'{self.folder_id}' in parents and trashed=false"
        results = self.service.files().list(
            q=query, 
            fields="nextPageToken, files(id, name)",
            pageSize=1000
        ).execute()
        files = results.get('files', [])
        
        dna_files = []
        for f in files:
            match = re.search(r'DNA成分股_(\d{8})\.xlsx', f['name'])
            if match:
                f['date_str'] = match.group(1)
                dna_files.append(f)
                
        return sorted(dna_files, key=lambda x: x['date_str'], reverse=True)

    def download_and_parse_excel(self, file_id: str) -> pd.DataFrame:
        """下載 Excel 內容至記憶體並轉為 DataFrame"""
        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            
        fh.seek(0)
        return pd.read_excel(fh, dtype=str, engine='openpyxl')

    # ==========================================
    # 總表 (Master Table) 處理 (上傳與下載)
    # ==========================================
    def get_master_table_info(self) -> dict:
        """尋找雲端資料夾中是否已有 DNA_Master_Table.csv"""
        query = f"'{self.folder_id}' in parents and name='DNA_Master_Table.csv' and trashed=false"
        results = self.service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get('files', [])
        return files[0] if files else None

    def download_csv_to_df(self, file_id: str) -> pd.DataFrame:
        """專門用來下載並解析 CSV 總表"""
        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        fh.seek(0)
        return pd.read_csv(fh, dtype=str, encoding='utf-8-sig')

    def upload_master_table(self, df: pd.DataFrame, file_id: str = None):
        """將最新的 DataFrame 上傳/覆蓋回 Google Drive"""
        csv_buffer = io.BytesIO()
        df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        csv_buffer.seek(0)
        
        media = MediaIoBaseUpload(csv_buffer, mimetype='text/csv', resumable=True)
        
        if file_id:
            self.service.files().update(fileId=file_id, media_body=media).execute()
            print("✅ 總表已成功更新至 Google Drive！")
        else:
            file_metadata = {'name': 'DNA_Master_Table.csv', 'parents': [self.folder_id]}
            self.service.files().create(body=file_metadata, media_body=media).execute()
            print("✅ 已建立全新的總表並上傳至 Google Drive！")

    # ==========================================
    # 核心邏輯：增量更新與條件篩選
    # ==========================================
    def get_eligible_dna_stocks(self) -> pd.DataFrame:
        """
        [資料前處理模組 V3 - 增量更新架構] 
        1. 讀取歷史總表 (若有)
        2. 比對日期，只下載最新的 Excel 檔
        3. 更新並上傳總表
        4. 依據總表計算連續條件與漲跌幅
        """
        # --- [階段一：準備總表與新資料] ---
        master_info = self.get_master_table_info()
        master_df = pd.DataFrame()
        latest_date_in_db = "00000000"
        
        if master_info:
            master_df = self.download_csv_to_df(master_info['id'])
            if not master_df.empty and '日期' in master_df.columns:
                latest_date_in_db = master_df['日期'].max()
                
        all_excel_files = self.fetch_dna_excel_files()
        if not all_excel_files:
            return pd.DataFrame()
            
        new_files = [f for f in all_excel_files if f['date_str'] > latest_date_in_db]
        
        if new_files:
            print(f"發現 {len(new_files)} 個新檔案，準備進行增量更新...")
            new_data_list = []
            for f_info in new_files:
                df_temp = self.download_and_parse_excel(f_info['id'])
                sid_col = next((c for c in df_temp.columns if '股號' in c or '代號' in c), '股號')
                name_col = next((c for c in df_temp.columns if '股名' in c or '名稱' in c), '股名')
                price_col = next((c for c in df_temp.columns if '股價' in c or '收盤' in c), '收盤價')
                
                for _, row in df_temp.iterrows():
                    sid = str(row.get(sid_col, '')).strip()
                    if not sid or sid == 'nan': continue
                    
                    price_val = str(row.get(price_col, '0')).replace(',', '')
                    try: current_price = float(price_val)
                    except ValueError: current_price = 0.0
                        
                    new_data_list.append({
                        '日期': f_info['date_str'],
                        '股號': sid,
                        '股名': str(row.get(name_col, '')).strip(),
                        '收盤價': current_price
                    })
            
            new_df = pd.DataFrame(new_data_list)
            master_df = pd.concat([master_df, new_df], ignore_index=True) if not master_df.empty else new_df
            self.upload_master_table(master_df, master_info['id'] if master_info else None)
        else:
            print("雲端無新檔案，直接使用現有總表進行運算。")

        # --- [階段二：根據最新總表計算連續條件與排行] ---
        if master_df.empty:
            return pd.DataFrame()

        all_dates = sorted(master_df['日期'].unique(), reverse=True)
        min_consecutive_days = 2
        missing_days_threshold = 2
        
        eligible_stocks = []
        grouped = master_df.groupby('股號')
        
        for sid, group in grouped:
            stock_dates = set(group['日期'].tolist())
            presence = [1 if d in stock_dates else 0 for d in all_dates]
            presence_str = "".join(map(str, presence))
            
            has_consecutive = ("1" * min_consecutive_days) in presence_str
            is_missing = all(p == 0 for p in presence[:missing_days_threshold]) if len(presence) >= missing_days_threshold else False
            
            if has_consecutive and not is_missing:
                group = group.sort_values('日期')
                old_p = group.iloc[0]['收盤價']
                file_new_p = group.iloc[-1]['收盤價']
                stock_name = group.iloc[-1]['股名']
                
                from data_engine import fetch_finmind_data
                df_live = fetch_finmind_data(sid, years=0.1)
                if not df_live.empty:
                    real_new_p = df_live['close'].iloc[-1]
                else:
                    real_new_p = file_new_p
                
                true_change_pct = ((real_new_p - old_p) / old_p * 100) if old_p != 0 else 0.0
                
                eligible_stocks.append({
                    '股號': sid,
                    '股名': stock_name,
                    '收盤價(最新日期)': real_new_p,  # 這裡是真實市價
                    '漲跌幅 (%)': round(true_change_pct, 2), # 這裡是真實漲跌幅
                    '最舊收盤價': old_p # 傳給下一手引擎備用
                })
                
        result_df = pd.DataFrame(eligible_stocks)
        if not result_df.empty:
            result_df = result_df.sort_values(by='漲跌幅 (%)', ascending=False).reset_index(drop=True)
            
        return result_df

# --- 本機單獨測試用 (可選) ---
if __name__ == "__main__":
    # 設定區
    CONFIG = {
        'KEY': 'credentials.json',
        'FOLDER_ID': '你的雲端資料夾ID'
    }

    print("啟動 Drive Data Engine 測試...")
    engine = DriveDataEngine(CONFIG['KEY'], CONFIG['FOLDER_ID'])
    final_list = engine.get_eligible_dna_stocks()
    print("產出結果：")
    print(final_list.head() if not final_list.empty else "名單為空")
