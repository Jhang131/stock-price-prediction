import sqlite3

# 設定資料庫檔案的路徑
db_path = r'C:\Users\OWNER\OneDrive\桌面\stock_price_pridiction\database\my_database.db'

# 連接到資料庫（如果不存在，則會創建一個新的資料庫）
conn = sqlite3.connect(db_path)

# 關閉資料庫連接
conn.close()

print("資料庫已成功創建並且連接已關閉")