import io
import os
import re
import pandas as pd
import numpy as np
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

class DriveDataEngine:
    def __init__(self, credentials_path: str, folder_id: str):
        """
        初始化 Google Drive 連線引擎
        :param credentials_path: JSON 金鑰的路徑 (例如 'credentials.json')
        :param folder_id: Google Drive 資料夾的 ID
        """
        self.credentials_path = credentials_path
        self.folder_id = folder_id
        self.scopes = ['https://www.googleapis.com/auth/drive.readonly']
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
        import os
        if os.path.exists(self.credentials_path):
            creds = service_account.Credentials.from_service_account_file(
                self.credentials_path, scopes=self.scopes)
            return build('drive', 'v3', credentials=creds)
            
        else:
            raise FileNotFoundError("⚠️ 找不到 API 憑證！請確認是否設定了 Streamlit Secrets 或 credentials.json。")

    def fetch_all_csv_files(self) -> list:
        """列出資料夾內所有非垃圾桶的 CSV 檔案"""
        query = f"'{self.folder_id}' in parents and mimeType='text/csv' and trashed=false"
        results = self.service.files().list(
            q=query, 
            fields="nextPageToken, files(id, name)",
            pageSize=1000
        ).execute()
        files = results.get('files', [])
        return sorted(files, key=lambda x: x['name']) # 依檔名初步排序

    def download_and_parse_csv(self, file_id: str, file_name: str) -> pd.DataFrame:
        """下載 CSV 內容至記憶體並轉為 DataFrame"""
        request = self.service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        
        done = False
        while done is False:
            status, done = downloader.next_chunk()
            
        fh.seek(0)
        
        # 處理台股常見的 Big5 編碼問題
        try:
            df = pd.read_csv(fh, dtype=str)
        except UnicodeDecodeError:
            fh.seek(0)
            df = pd.read_csv(fh, encoding='big5', errors='replace', dtype=str)
            
        df['Source_File_Name'] = file_name 
        return df

    def build_master_dataframe(self) -> pd.DataFrame:
        """
        [核心引擎] 自動彙整所有檔案、萃取日期、清洗資料並計算總漲幅
        """
        files = self.fetch_all_csv_files()
        if not files:
            print("⚠️ 雲端資料夾內無 CSV 檔案。")
            return pd.DataFrame()

        all_dfs = []
        for file_info in files:
            df = self.download_and_parse_csv(file_info['id'], file_info['name'])
            if not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            return pd.DataFrame()

        # 1. 垂直合併所有日期資料
        master_df = pd.concat(all_dfs, ignore_index=True)
        
        # 2. 從檔名萃取日期 (YYYYMMDD)
        master_df['Date_Str'] = master_df['Source_File_Name'].apply(
            lambda x: re.search(r'\d{8}', x).group(0) if re.search(r'\d{8}', x) else None
        )
        master_df['Date'] = pd.to_datetime(master_df['Date_Str'], format='%Y%m%d', errors='coerce')
        
        # 3. 清洗數值欄位 (處理千分位逗號與非數字符號)
        for col in ['收盤價']:
            if col in master_df.columns:
                master_df[col] = master_df[col].astype(str).str.replace(',', '', regex=False)
                master_df[col] = pd.to_numeric(master_df[col], errors='coerce')
        
        # 剔除無效數據
        master_df = master_df.dropna(subset=['證券代號', '收盤價', 'Date'])
        
        # 4. 依照股號分組計算漲幅
        summary = self.calculate_returns(master_df)
        return summary

    def fetch_dna_excel_files(self) -> list:
        """列出資料夾內所有符合 DNA成分股_YYYYMMDD.xlsx 的檔案，由新到舊排序"""
        # 抓取資料夾內非垃圾桶的檔案
        query = f"'{self.folder_id}' in parents and trashed=false"
        results = self.service.files().list(
            q=query, 
            fields="nextPageToken, files(id, name)",
            pageSize=1000
        ).execute()
        files = results.get('files', [])
        
        # 篩選檔名並萃取日期
        dna_files = []
        for f in files:
            # 支援 .xlsx (使用正規表達式抓取日期)
            match = re.search(r'DNA成分股_(\d{8})\.xlsx', f['name'])
            if match:
                f['date_str'] = match.group(1)
                dna_files.append(f)
                
        # 依日期由新到舊排序 (遞減)
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
        # 注意：需要確保環境中有安裝 openpyxl (pip install openpyxl)
        df = pd.read_excel(fh, dtype=str, engine='openpyxl')
        return df

    def get_eligible_dna_stocks(self) -> pd.DataFrame:
        """
        [資料前處理模組 V2] 
        1. 依日期抓取所有 DNA Excel 檔案
        2. 計算出現次數、連續條件，並計算區間漲跌幅
        3. 輸出：股號、股名、最新收盤價、漲跌幅 (%) (由高到低排行)
        """
        files = self.fetch_dna_excel_files() # 這裡已經是由新到舊排序
        if not files:
            print("⚠️ 雲端資料夾內無符合條件的 DNA成分股 Excel 檔案。")
            return pd.DataFrame()
            
        # 邏輯參數
        min_consecutive_days = 2
        missing_days_threshold = 2
            
        stock_history = {}
        
        # 遍歷檔案 (i=0 是最新的一天)
        for i, f_info in enumerate(files):
            df = self.download_and_parse_excel(f_info['id'])
            
            # 自動定位欄位名稱
            sid_col = next((c for c in df.columns if '股號' in c or '代號' in c), '股號')
            name_col = next((c for c in df.columns if '股名' in c or '名稱' in c), '股名')
            price_col = next((c for c in df.columns if '股價' in c or '收盤' in c), '收盤價')
            
            for _, row in df.iterrows():
                sid = str(row.get(sid_col, '')).strip()
                if not sid or sid == 'nan': continue
                
                # 清洗價格數據 (處理千分位逗號)
                price_val = str(row.get(price_col, '0')).replace(',', '')
                try:
                    current_price = float(price_val)
                except ValueError:
                    current_price = 0.0

                if sid not in stock_history:
                    # 第一次遇到 (最新日期)
                    stock_history[sid] = {
                        '股號': sid,
                        '股名': str(row.get(name_col, '')).strip(),
                        '最新收盤價': current_price,
                        '最舊收盤價': current_price, # 預設最舊也是這天的價格
                        'presence': [0] * len(files)
                    }
                
                # 只要後續在更舊的檔案中有出現，就更新「最舊收盤價」
                stock_history[sid]['最舊收盤價'] = current_price
                stock_history[sid]['presence'][i] = 1

        # 篩選與排行
        eligible_stocks = []
        for sid, data in stock_history.items():
            presence = data['presence']
            presence_str = "".join(map(str, presence))
            
            # 條件判斷
            has_consecutive = ("1" * min_consecutive_days) in presence_str
            is_missing = all(p == 0 for p in presence[:missing_days_threshold]) if len(presence) >= missing_days_threshold else False
            
            if has_consecutive and not is_missing:
                # 計算漲跌幅 %
                old_p = data['最舊收盤價']
                new_p = data['最新收盤價']
                change_pct = ((new_p - old_p) / old_p * 100) if old_p != 0 else 0.0
                
                eligible_stocks.append({
                    '股號': data['股號'],
                    '股名': data['股名'],
                    '收盤價(最新日期)': new_p,
                    '漲跌幅 (%)': round(change_pct, 2)
                })
        
        result_df = pd.DataFrame(eligible_stocks)
        
        if not result_df.empty:
            # 按漲跌幅由高到低排列
            result_df = result_df.sort_values(by='漲跌幅 (%)', ascending=False).reset_index(drop=True)
            
        return result_df

    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        計算每檔股票在現有資料區間內的總漲幅
        """
        # 確保按日期排序
        df = df.sort_values(by=['證券代號', 'Date'])
        
        # 聚合計算：取第一個日期的價格與最後一個日期的價格
        summary = df.groupby('證券代號').agg(
            最舊收盤價=('收盤價', 'first'),
            最新收盤價=('收盤價', 'last'),
            證券名稱=('證券名稱', 'last'),
            開始日期=('Date_Str', 'first'),
            結束日期=('Date_Str', 'last')
        ).reset_index()

        # 計算總漲幅百分比
        summary['總漲幅 (%)'] = ((summary['最新收盤價'] - summary['最舊收盤價']) / summary['最舊收盤價']) * 100
        summary['總漲幅 (%)'] = summary['總漲幅 (%)'].round(2)
        
        # 欄位重新命名以符合 UI 需求
        summary = summary.rename(columns={
            '證券代號': '股號', 
            '最新收盤價': '當日收盤價'
        })
        
        # 回傳最終清單
        return summary[['股號', '證券名稱', '當日收盤價', '總漲幅 (%)', '開始日期', '結束日期']]

# --- 使用範例 ---
if __name__ == "__main__":
    # 設定區
    CONFIG = {
        'KEY': 'credentials.json',
        'FOLDER_ID': '你的雲端資料夾ID'
    }

    engine = DriveDataEngine(CONFIG['KEY'], CONFIG['FOLDER_ID'])
    final_list = engine.build_master_dataframe()
    print(final_list.head())
