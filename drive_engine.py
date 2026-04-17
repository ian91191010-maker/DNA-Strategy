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