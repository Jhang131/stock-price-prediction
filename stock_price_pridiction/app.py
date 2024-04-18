from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense
import matplotlib.gridspec as gridspec
import sqlite3
from bs4 import BeautifulSoup
import requests

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        stock_code = request.form['stock_code']
        return predict_stock(stock_code)

    return render_template('index.html')


def search(stock_code):
    url = f'https://goodinfo.tw/tw/StockDetail.asp?STOCK_ID={stock_code}'
    header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    response = requests.get(url, headers=header)
    html = response.content.decode('utf-8')
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table', {'class': 'b1 p4_4 r10 box_shadow'})

    ch_text = []
    for row in table.find_all('tr'):
        for cell in row.find_all('td'):
            if cell.text.strip():
                ch_text.append(cell.text.strip())

    for i in range(len(ch_text)-1):
        if ch_text[i] == '市/ 櫃':
            return ch_text[i+1]
        

def predict_stock(stock_code):
    url = f'https://goodinfo.tw/tw/StockDetail.asp?STOCK_ID={stock_code}'
    header = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
    response = requests.get(url, headers=header)
    html = response.content.decode('utf-8')
    soup = BeautifulSoup(html, 'html.parser')
    table = soup.find('table', {'class': 'b1 p4_4 r10 box_shadow'})
    
    ch_text = []
    for row in table.find_all('tr'):
        for cell in row.find_all('td'):
            if cell.text.strip():
                ch_text.append(cell.text.strip())

    for i in range(len(ch_text)-1):
        if ch_text[i] == '市/ 櫃':
            return ch_text[i+1]
        return render_template('result.html', stock_code=stock_code)

# 讀取數據
while True:
    stock_code = input('請輸入股票代號:')
    if len(stock_code) == 4 and stock_code.isdigit():
        break
    else:
        print('請輸入台股四位數代號')
        stock_code = input('請輸入股票代號:')
stock = stock_code + '.TW'
symbol = yf.download(stock, start='2023-01-01',end='2023-12-31', progress=False)
data = symbol[['Close']]

# 數據預處理
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)

# 創建時間序列數據集
def create_time_series_dataset(dataset, time_steps):
    data_X, data_Y = [], []
    for i in range(len(dataset) - time_steps):
        a = dataset[i:(i + time_steps), 0]
        data_X.append(a)
        data_Y.append(dataset[i + time_steps, 0])
    return np.array(data_X), np.array(data_Y)

# 使用過去30天的數據
time_steps = 30
X, Y = create_time_series_dataset(data_scaled, time_steps)

# 划分訓練集和測試集
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
trainX, trainY = X[0:train_size, :], Y[0:train_size]
testX, testY = X[train_size:len(X), :], Y[train_size:len(Y)]

# 調整數據形狀以滿足LSTM模型輸入要求
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# # 構建LSTM模型
# model = Sequential()
# model.add(LSTM(50, input_shape=(1, time_steps)))
# model.add(Dense(1))
# model.compile(optimizer='adam', loss='mean_squared_error')

# # 訓練模型
# model.fit(trainX, trainY, epochs=50, batch_size=1, verbose=2)
# target_path = r'/Users/jiaru/Desktop/stock/model/'
# model.save(f"{target_path}\lstm_model.h5")

# # 預測
# trainPredict = model.predict(trainX)
# testPredict = model.predict(testX)

# 加载模型
target_path = r'C:\Users\OWNER\OneDrive\桌面\stock_price_pridiction\model'
loaded_model = load_model(f"{target_path}\lstm_model.h5")

# 預測 (載入模型)
trainPredict = loaded_model.predict(trainX)
testPredict = loaded_model.predict(testX)

# 反歸一化
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# 將股價數據反歸一化
original_data = scaler.inverse_transform(data_scaled)
# 未來30天的預測示例
last_30_days = data_scaled[-time_steps:]
future_dates = [data.index[-1] + pd.DateOffset(i) for i in range(1, 31)]

# 將最後30天的數據轉換為模型所需的形狀
input_data = np.reshape(last_30_days, (1, 1, time_steps))
future_predictions = []

# 逐步預測未來30天的股價
for i in range(30):
    future_prediction = loaded_model.predict(input_data)
    future_predictions.append(future_prediction[0, 0])

    # 更新input_data，加入新預測的值
    input_data = np.append(input_data[:, :, 1:], future_prediction[0, 0])
    input_data = np.reshape(input_data, (1, 1, time_steps))

# 反歸一化未來預測數據
future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))



###___COMBINED___###



# 生成買賣訊號
df_combined = data.copy()
short_window_ma = 5
long_window_ma = 20

signals_ma = pd.DataFrame(index=df_combined.index)
signals_ma['signal_ma'] = 0.0

signals_ma['short_ma'] = df_combined['Close'].rolling(window=short_window_ma, min_periods=1, center=False).mean()
signals_ma['long_ma'] = df_combined['Close'].rolling(window=long_window_ma, min_periods=1, center=False).mean()

# 當短期均線上穿長期均線時，生成買入訊號為1.0
signals_ma['signal_ma'][short_window_ma:] = np.where((signals_ma['short_ma'][short_window_ma:]
                                            > signals_ma['long_ma'][short_window_ma:])& (df_combined['Close'][short_window_ma:] > signals_ma['long_ma'][short_window_ma:]), 1.0, 0.0)
# 當收盤價跌破長期均線時，生成賣出訊號為-1.0
# signals_ma['signal_ma'][short_window_ma:] = np.where(df_combined['Close'][short_window_ma:] < signals_ma['long_ma'][short_window_ma:], -1.0,signals_ma['signal_ma'][short_window_ma:])

signals_ma['positions_ma'] = signals_ma['signal_ma'].diff()

# 生成買賣訊號 (基於MACD)
short_window_macd = 12  # MACD短週期
long_window_macd = 26  # MACD長週期

# 計算MACD指標
exp12 = df_combined['Close'].ewm(span=short_window_macd, adjust=False).mean()
exp26 = df_combined['Close'].ewm(span=long_window_macd, adjust=False).mean()
macd = exp12 - exp26
signal_macd = macd.ewm(span=9, adjust=False).mean()

# 產生買賣訊號
signals_macd = pd.DataFrame(index=df_combined.index)
signals_macd['signal_macd'] = 0.0
signals_macd['macd'] = macd
signals_macd['signal_macd'][short_window_macd:] = np.where(signals_macd['macd'][short_window_macd:] > 0, 1.0, 0.0)
signals_macd['positions_macd'] = signals_macd['signal_macd'].diff()


# 合併兩個訊號
signals_combined = pd.concat([signals_ma['signal_ma'], signals_macd['signal_macd']], axis=1)


signals_combined['combined_signal'] = np.where((signals_ma['short_ma'] > signals_ma['long_ma'])& (df_combined['Close'][short_window_ma:] > signals_ma['long_ma'][short_window_ma:]) & (signals_combined['signal_macd'] > 0), 1.0, 0.0)
# 當收盤價跌破長期均線時，生成賣出訊號為-1.0
# signals_combined['combined_signal'] = np.where((df_combined['Close'][short_window_ma:] < signals_ma['long_ma'][short_window_ma:]) & (signals_combined['signal_macd'] > 0), -1.0, signals_combined['combined_signal'])

signals_combined['combined_positions'] = signals_combined['combined_signal'].diff()

df_combined = pd.concat([df_combined, signals_combined[['combined_signal', 'combined_positions']]], axis=1)

# 均線訊號
signals_combined['short_5Dma'] = df_combined['Close'].rolling(window=5, min_periods=1, center=False).mean()
signals_combined['medium_20Dma'] = df_combined['Close'].rolling(window=20, min_periods=1, center=False).mean()
signals_combined['long_60Dma'] = df_combined['Close'].rolling(window=60, min_periods=1, center=False).mean()
signals_combined['extra_180Dma'] = df_combined['Close'].rolling(window=180, min_periods=1, center=False).mean()


def buy_stock(real_movement, signal, initial_money=10000, max_buy=1, max_sell=1):
    starting_money = initial_money
    states_sell = []
    states_buy = []
    current_inventory = 0

    def buy(i, initial_money, current_inventory):
        shares = initial_money // real_movement[i]
        if shares < 1:
            print('day %d: total balances %f, not enough money to buy a unit price %f' % (i, initial_money, real_movement[i]))
        else:
            if shares > max_buy:
                buy_units = max_buy
            else:
                buy_units = shares
            initial_money -= buy_units * real_movement[i]
            current_inventory += buy_units
            print('day %d: buy %d units at price %f, total balance %f' % (i, buy_units, buy_units * real_movement[i], initial_money))
            states_buy.append(i)
        return initial_money, current_inventory

    # def sell(i, initial_money, current_inventory):
    #     if current_inventory == 0:
    #         print('day %d: cannot sell anything, inventory 0' % (i))
    #     else:
    #         if current_inventory > max_sell:
    #             sell_units = max_sell
    #         else:
    #             sell_units = current_inventory
    #         current_inventory -= sell_units
    #         total_sell = sell_units * real_movement[i]
    #         initial_money += total_sell
    #         try:
    #             invest = ((real_movement[i] - real_movement[states_buy[-1]]) / real_movement[states_buy[-1]]) * 100
    #         except:
    #             invest = 0
    #         print('day %d, sell %d units at price %f, investment %f %%, total balance %f,' % (i, sell_units, total_sell, invest, initial_money))
    #         states_sell.append(i)
    #     return initial_money, current_inventory

    for i in range(real_movement.shape[0] - 20):
        state = signal[i]
        if i >= 20 and real_movement[i] < np.mean(real_movement[i-19:i+1]):

            if current_inventory > 0:
                print('day %d: stock price is below 20-day average, selling all units' % i)
                current_inventory = 0
                states_sell.append(i)
            continue   
        if state == 1:
            initial_money, current_inventory = buy(i, initial_money, current_inventory)
            states_buy.append(i)
        elif state == -1:
            if current_inventory == 0:
                print('day %d: cannot sell anything, inventory 0' % (i))
            else:
                if current_inventory > max_sell:
                    sell_units = max_sell
                else:
                    sell_units = current_inventory
                current_inventory -= sell_units
                total_sell = sell_units * real_movement[i]
                initial_money += total_sell
                try:
                    invest = ((real_movement[i] - real_movement[states_buy[-1]]) / real_movement[states_buy[-1]]) * 100
                except:
                    invest = 0
                print('day %d, sell %d units at price %f, investment %f %%, total balance %f,' % (i, sell_units, total_sell, invest, initial_money))
            states_sell.append(i)
    invest = ((initial_money - starting_money) / starting_money) * 100
    
    return states_buy, states_sell, invest

# 獲取買賣訊號
states_buy, states_sell, invest = buy_stock(df_combined.Close, signals_combined['combined_positions'])



###___MA___###



# 生成買賣訊號
df_ma = data.copy()
ma_short_window = 5

ma_long_window = 20


ma_signals = pd.DataFrame(index=df_ma.index)
ma_signals['signal'] = 0.0

ma_signals['short_ma'] = df_ma['Close'].rolling(window=ma_short_window, min_periods=1, center=False).mean()
ma_signals['long_ma'] = df_ma['Close'].rolling(window=ma_long_window, min_periods=1, center=False).mean()

# 當短期均線上穿長期均線時，生成買入訊號為1.0
ma_signals['signal'][ma_short_window:] = np.where((ma_signals['short_ma'][ma_short_window:]
                                            > ma_signals['long_ma'][ma_short_window:])& (df_ma['Close'][ma_short_window:] > ma_signals['long_ma'][ma_short_window:]), 1.0, 0.0)
# 當收盤價跌破長期均線時，生成賣出訊號為-1.0
# ma_signals['signal'][ma_short_window:] = np.where(df_ma['Close'][ma_short_window:] < ma_signals['long_ma'][ma_short_window:],-1.0,ma_signals['signal'][ma_short_window:])

ma_signals['ma_positions'] = ma_signals['signal'].diff()

# 均線訊號
ma_signals['short_5Dma'] = data['Close'].rolling(window=5, min_periods=1, center=False).mean()
ma_signals['medium_20Dma'] = data['Close'].rolling(window=20, min_periods=1, center=False).mean()
ma_signals['long_60Dma'] = data['Close'].rolling(window=60, min_periods=1, center=False).mean()
ma_signals['extra_180Dma'] = data['Close'].rolling(window=180, min_periods=1, center=False).mean()

def buy_stock_ma(real_movement, signal, initial_money=10000, max_buy=1, max_sell=1):
    starting_money = initial_money
    states_sell_ma = []
    states_buy_ma = []
    current_inventory = 0

    def buy(i, initial_money, current_inventory):
        shares = initial_money // real_movement[i]
        if shares < 1:
            print('day %d: total balances %f, not enough money to buy a unit price %f' % (i, initial_money, real_movement[i]))
        else:
            if shares > max_buy:
                buy_units = max_buy
            else:
                buy_units = shares
            initial_money -= buy_units * real_movement[i]
            current_inventory += buy_units
            print('day %d: buy %d units at price %f, total balance %f' % (i, buy_units, buy_units * real_movement[i], initial_money))
            states_buy_ma.append(i)
        return initial_money, current_inventory

    for i in range(real_movement.shape[0] - int(0.025 * len(df_ma))):
        state = signal[i]
        if state == 1:
            initial_money, current_inventory = buy(i, initial_money, current_inventory)
            states_buy_ma.append(i)
        elif state == -1:
            if current_inventory == 0:
                print('day %d: cannot sell anything, inventory 0' % (i))
            else:
                if current_inventory > max_sell:
                    sell_units = max_sell
                else:
                    sell_units = current_inventory
                current_inventory -= sell_units
                total_sell = sell_units * real_movement[i]
                initial_money += total_sell
                try:
                    invest_ma = ((real_movement[i] - real_movement[states_buy_ma[-1]]) / real_movement[states_buy_ma[-1]]) * 100
                except:
                    invest_ma = 0
                print('day %d, sell %d units at price %f, investment %f %%, total balance %f,' % (i, sell_units, total_sell, invest, initial_money))
            states_sell_ma.append(i)
    invest_ma = ((initial_money - starting_money) / starting_money) * 100

    return states_buy_ma, states_sell_ma, invest_ma

# 獲取買賣訊號
states_buy_ma, states_sell_ma, invest_ma = buy_stock_ma(df_ma.Close, ma_signals['ma_positions'])



###___MACD___###



# 生成買賣訊號 (基於MACD)
df_macd = data.copy()
macd_short_window = 12  # MACD短週期
macd_long_window = 26  # MACD長週期

# 計算MACD指標
exp12 = df_macd['Close'].ewm(span=macd_short_window, adjust=False).mean()
exp26 = df_macd['Close'].ewm(span=macd_long_window, adjust=False).mean()
macd = exp12 - exp26
macd_signal = macd.ewm(span=9, adjust=False).mean()

# 產生買賣訊號
macd_signals = pd.DataFrame(index=df_macd.index)
macd_signals['macd_signal'] = 0.0
macd_signals['macd'] = macd

# 當MACD翻正時，生成買入訊號為1.0，反之為賣出
macd_signals['macd_signal'][macd_short_window:] = np.where(macd_signals['macd'][macd_short_window:] > 0, 1.0, 0.0)

macd_signals['macd_positions'] = macd_signals['macd_signal'].diff()

# 均線訊號
macd_signals['short_5Dma'] = data['Close'].rolling(window=5, min_periods=1, center=False).mean()
macd_signals['medium_20Dma'] = data['Close'].rolling(window=20, min_periods=1, center=False).mean()
macd_signals['long_60Dma'] = data['Close'].rolling(window=60, min_periods=1, center=False).mean()
macd_signals['extra_180Dma'] = data['Close'].rolling(window=180, min_periods=1, center=False).mean()

def buy_stock_macd(real_movement, signal, initial_money=10000, max_buy=1, max_sell=1):
    starting_money = initial_money
    states_sell_macd = []
    states_buy_macd = []
    current_inventory = 0

    def buy(i, initial_money, current_inventory):
        shares = initial_money // real_movement[i]
        if shares < 1:
            print('day %d: total balances %f, not enough money to buy a unit price %f' % (i, initial_money, real_movement[i]))
        else:
            if shares > max_buy:
                buy_units = max_buy
            else:
                buy_units = shares
            initial_money -= buy_units * real_movement[i]
            current_inventory += buy_units
            print('day %d: buy %d units at price %f, total balance %f' % (i, buy_units, buy_units * real_movement[i], initial_money))
            states_buy_macd.append(i)
        return initial_money, current_inventory

    # def sell(i, initial_money, current_inventory):
    #     if current_inventory == 0:
    #         print('day %d: cannot sell anything, inventory 0' % (i))
    #     else:
    #         if current_inventory > max_sell:
    #             sell_units = max_sell
    #         else:
    #             sell_units = current_inventory
    #         current_inventory -= sell_units
    #         total_sell = sell_units * real_movement[i]
    #         initial_money += total_sell
    #         try:
    #             invest = ((real_movement[i] - real_movement[states_buy_macd[-1]]) / real_movement[states_buy_macd[-1]]) * 100
    #         except:
    #             invest = 0
    #         print('day %d, sell %d units at price %f, investment %f %%, total balance %f,' % (i, sell_units, total_sell, invest, initial_money))
    #         states_sell_macd.append(i)
    #     return initial_money, current_inventory
    
    for i in range(real_movement.shape[0] - int(0.025 * len(df_macd))):
        state = signal[i]
        if state == 1:
            initial_money, current_inventory = buy(i, initial_money, current_inventory)
            states_buy_macd.append(i)
        elif state == -1:
            if current_inventory == 0:
                print('day %d: cannot sell anything, inventory 0' % (i))
            else:
                if current_inventory > max_sell:
                    sell_units = max_sell
                else:
                    sell_units = current_inventory
                current_inventory -= sell_units
                total_sell = sell_units * real_movement[i]
                initial_money += total_sell
                try:
                    invest_macd = ((real_movement[i] - real_movement[states_buy_macd[-1]]) / real_movement[states_buy_macd[-1]]) * 100
                except:
                    invest_macd = 0
                print('day %d, sell %d units at price %f, investment %f %%, total balance %f,' % (i, sell_units, total_sell, invest, initial_money))
            states_sell_macd.append(i)
    invest_macd = ((initial_money - starting_money) / starting_money) * 100

    return states_buy_macd, states_sell_macd, invest_macd

# 獲取買賣訊號
states_buy_macd, states_sell_macd, invest_macd = buy_stock_macd(df_macd.Close, macd_signals['macd_positions'])


###MACD翻正＋五日沒新高賣出###

# # Generate buy/sell signals (based on MACD)
# df_5 = data.copy()
# short_macd = 12
# long_macd = 26

# # Calculate MACD indicator
# exp12 = df_5['Close'].ewm(span=short_macd, adjust=False).mean()
# exp26 = df_5['Close'].ewm(span=long_macd, adjust=False).mean()
# macd = exp12 - exp26
# signal_5 = macd.ewm(span=9, adjust=False).mean()

# # Generate buy/sell signals
# signal_5s = pd.DataFrame(index=df_5.index)
# signal_5s['signal_5'] = 0.0
# signal_5s['macd'] = macd
# signal_5s['signal_5'][short_macd:] = np.where(signal_5s['macd'][short_macd:] > 0, 1.0, 0.0)
# signal_5s['positions_5'] = signal_5s['signal_5'].diff()

# # 均線訊號
# signal_5s['short_5Dma'] = data['Close'].rolling(window=5, min_periods=1, center=False).mean()
# signal_5s['medium_20Dma'] = data['Close'].rolling(window=20, min_periods=1, center=False).mean()
# signal_5s['long_60Dma'] = data['Close'].rolling(window=60, min_periods=1, center=False).mean()
# signal_5s['extra_180Dma'] = data['Close'].rolling(window=180, min_periods=1, center=False).mean()

# def buy_stock_macd_5(real_movement, signal, initial_money=10000, max_buy=1, max_sell=1):
#     starting_money = initial_money
#     states_sell_macd_5 = []
#     states_buy_macd_5 = []
#     current_inventory = 0

#     def buy(i, initial_money, current_inventory):
#         shares = initial_money // real_movement[i]
#         if shares < 1:
#             print('day %d: total balances %f, not enough money to buy a unit price %f' % (i, initial_money, real_movement[i]))
#         else:
#             if shares > max_buy:
#                 buy_units = max_buy
#             else:
#                 buy_units = shares
#             initial_money -= buy_units * real_movement[i]
#             current_inventory += buy_units
#             print('day %d: buy %d units at price %f, total balance %f' % (i, buy_units, buy_units * real_movement[i], initial_money))
#             states_buy_macd_5.append(i)
#         return initial_money, current_inventory

#     def sell(i, initial_money, current_inventory):
#         if current_inventory == 0:
#             print('day %d: cannot sell anything, inventory 0' % (i))
#         else:
#             if current_inventory > max_sell:
#                 sell_units = max_sell
#             else:
#                 sell_units = current_inventory
#             current_inventory -= sell_units
#             total_sell = sell_units * real_movement[i]
#             initial_money += total_sell
#             try:
#                 invest_macd_5 = ((real_movement[i] - real_movement[states_buy_macd_5[-1]]) / real_movement[states_buy_macd_5[-1]]) * 100
#             except:
#                 invest_macd_5 = 0
#             print('day %d, sell %d units at price %f, investment %f %%, total balance %f,' % (i, sell_units, total_sell, invest_macd_5, initial_money))
#             states_sell_macd_5.append(i)
#         return initial_money, current_inventory

#     for i in range(real_movement.shape[0] - int(0.025 * len(df_5))):
#         state = signal[i]
#         if state == 1:
#             initial_money, current_inventory = buy(i, initial_money, current_inventory)
#             states_buy_macd_5.append(i)
#         elif state == -1:
#             # Modify the selling part
#             if i + 5 < len(df_5) and max(real_movement[i:i+5]) <= real_movement[i]:
#                 initial_money, current_inventory = sell(i, initial_money, current_inventory)
#             else:
#                 initial_money, current_inventory = sell(i, initial_money, current_inventory)

#     invest_macd_5 = ((initial_money - starting_money) / starting_money) * 100
#     return states_buy_macd_5, states_sell_macd_5, invest_macd_5


# def buy_stock_macd_5(real_movement, signal, initial_money=10000, max_buy=1, max_sell=1):
#     starting_money = initial_money
#     states_sell_macd_5 = []
#     states_buy_macd_5 = []
#     current_inventory = 0

#     def buy(i, initial_money, current_inventory):
#         shares = initial_money // real_movement[i]
#         if shares < 1:
#             print('day %d: total balance %f, not enough money to buy a unit price %f' % (i, initial_money, real_movement[i]))
#         else:
#             if shares > max_buy:
#                 buy_units = max_buy
#             else:
#                 buy_units = shares
#             initial_money -= buy_units * real_movement[i]
#             current_inventory += buy_units
#             print('day %d: buy %d units at price %f, total balance %f' % (i, buy_units, buy_units * real_movement[i], initial_money))
#             states_buy_macd_5.append(i)
#         return initial_money, current_inventory

#     def sell(i, initial_money, current_inventory):
#         if current_inventory == 0:
#             print('day %d: cannot sell anything, inventory 0' % (i))
#         else:
#             if current_inventory > max_sell:
#                 sell_units = max_sell
#             else:
#                 sell_units = current_inventory
#             current_inventory -= sell_units
#             total_sell = sell_units * real_movement[i]
#             initial_money += total_sell
#             try:
#                 invest_macd_5 = ((real_movement[i] - real_movement[states_buy_macd_5[-1]]) / real_movement[states_buy_macd_5[-1]]) * 100
#             except:
#                 invest_macd_5 = 0
#             print('day %d, sell %d units at price %f, investment %f %%, total balance %f,' % (i, sell_units, total_sell, invest_macd_5, initial_money))
#             states_sell_macd_5.append(i)
#         return initial_money, current_inventory
    
#     for i in range(real_movement.shape[0] - int(0.025 * len(df_5))):
#         state = signal[i]

#         if state == 1:
#             initial_money, current_inventory = buy(i, initial_money, current_inventory)
#             states_buy_macd_5.append(i)
#         elif state == -1:
#             # Modify the selling part
#             if i + 5 < len(df_5) and max(real_movement[i:i+5]) <= real_movement[i]:
#                 initial_money, current_inventory = sell(i, initial_money, current_inventory)
#             else:
#                 initial_money, current_inventory = sell(i, initial_money, current_inventory)
# def buy_stock_macd_5(real_movement, signal, initial_money=10000, max_buy=1, max_sell=1):
#     starting_money = initial_money
#     states_sell_macd_5 = []
#     states_buy_macd_5 = []
#     current_inventory = 0

#     def buy(i, initial_money, current_inventory):
#         shares = initial_money // real_movement[i]
#         if shares < 1:
#             print('day %d: total balances %f, not enough money to buy a unit price %f' % (i, initial_money, real_movement[i]))
#         else:
#             if shares > max_buy:
#                 buy_units = max_buy
#             else:
#                 buy_units = shares
#             initial_money -= buy_units * real_movement[i]
#             current_inventory += buy_units
#             print('day %d: buy %d units at price %f, total balance %f' % (i, buy_units, buy_units * real_movement[i], initial_money))
#             states_buy_macd_5.append(i)
#         return initial_money, current_inventory

    # def sell(i, initial_money, current_inventory):
    #     if current_inventory == 0:
    #         print('day %d: cannot sell anything, inventory 0' % (i))
    #     else:
    #         if current_inventory > max_sell:
    #             sell_units = max_sell
    #         else:
    #             sell_units = current_inventory
    #         current_inventory -= sell_units
    #         total_sell = sell_units * real_movement[i]
    #         initial_money += total_sell
    #         try:
    #             invest = ((real_movement[i] - real_movement[states_buy_macd_5[-1]]) / real_movement[states_buy_macd_5[-1]]) * 100
    #         except:
    #             invest = 0
    #         print('day %d, sell %d units at price %f, investment %f %%, total balance %f,' % (i, sell_units, total_sell, invest, initial_money))
    #         states_sell_macd_5.append(i)
    #     return initial_money, current_inventory


    # for i in range(real_movement.shape[0] - 5):
    #     state = signal[i]
 
    #     if i >= 5 and real_movement[i] < np.mean(real_movement[i-4:i+1]):

    #         if current_inventory > 0:
    #             print('day %d: stock price is below 20-day average, selling all units' % i)
    #             current_inventory = 0
    #             states_sell_macd_5.append(i)
    #         continue   
        # # 初始化變數
        # buy_price = 0
        # days_since_purchase = 0        
        # # 在迴圈中加入以下條件
        # if current_inventory > 0:
        #     days_since_purchase += 1
            
        #     # 判斷是否滿五天
        #     if days_since_purchase >= 5:
        #         # 檢查過去五天中是否有任一天的價格高於買入價格
        #         if all(real_movement[i - j] <= buy_price for j in range(1, 6)):
        #             print('day %d: selling all units at price %.2f' % (i, real_movement[i]))
        #             current_inventory = 0
        #             days_since_purchase = 0
        #             states_sell_macd_5.append(i)
        #             continue

        #     if real_movement[i] > buy_price:
        #         # 當天價格高於買入價格，將其作為新的買入價格
        #         buy_price = real_movement[i]
        #         days_since_purchase = 0
        #         print('day %d: updating buy price to %.2f' % (i, buy_price))
        #     else:
        #         # 未滿五天或收盤價未高於買入價格，繼續持有
        #         continue
    #     if state == 1:
    #         initial_money, current_inventory = buy(i, initial_money, current_inventory)
    #         states_buy_macd_5.append(i)
    #     elif state == -1:
    #         if current_inventory == 0:
    #             print('day %d: cannot sell anything, inventory 0' % (i))
    #         else:
    #             if current_inventory > max_sell:
    #                 sell_units = max_sell
    #             else:
    #                 sell_units = current_inventory
    #             current_inventory -= sell_units
    #             total_sell = sell_units * real_movement[i]
    #             initial_money += total_sell
    #             try:
    #                 invest_macd_5 = ((real_movement[i] - real_movement[states_buy_macd_5[-1]]) / real_movement[states_buy_macd_5[-1]]) * 100
    #             except:
    #                 invest_macd_5 = 0
    #             print('day %d, sell %d units at price %f, investment %f %%, total balance %f,' % (i, sell_units, total_sell, invest, initial_money))
    #         states_sell_macd_5.append(i)

    # invest_macd_5 = ((initial_money - starting_money) / starting_money) * 100
  
  
    # return states_buy_macd_5, states_sell_macd_5, invest_macd_5



# Get buy/sell signals
# states_buy_macd_5, states_sell_macd_5, invest_macd_5 = buy_stock_macd_5(df_5.Close, signal_5s['positions_5'])



## 圖形合併
fig = plt.figure(figsize=(20, 30))

# 使用gridspec进行手动布局
gs = gridspec.GridSpec(3, 1, height_ratios=[1, 1, 1])

# 遍历每张小图，设置 X 轴刻度
for i in range(3):
    axes = plt.subplot(gs[i, 0])
    axes.xaxis.set_major_locator(mdates.MonthLocator())
    axes.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))




###___COMBINED圖形可視化___###


# 第一张图（放置在第一行的第一列）
axes1 = plt.subplot(gs[0, 0])
axes1.xaxis.set_major_locator(mdates.MonthLocator())
axes1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
# 股價走勢
axes1.plot(df_combined.index, original_data, label='Original Stock Price', linestyle='solid', color='black', alpha=0.5)


# 實際訓練和預測測試
axes1.plot(df_combined.index[time_steps:len(trainPredict) + time_steps], trainY.flatten(), label='Actual Train', color='blue')
axes1.plot(df_combined.index[len(trainPredict) + time_steps:], testPredict.flatten(), label='Predicted Test', color='orange')

# 繪製未來5天的預測
axes1.plot(future_dates, future_predictions, marker='o', markersize=2, label='Future Prediction', color='green')

# 計算並繪製買賣訊號
states_buy, states_sell, invest = buy_stock(df_combined['Close'], signals_combined['combined_positions'])
axes1.plot(df_combined.index[states_buy], original_data[states_buy], '^', markersize=10, color='m', label='Buy Signal')
axes1.plot(df_combined.index[states_sell], original_data[states_sell], 'v', markersize=10, color='k', label='Sell Signal')

# 計算總收益和投資報酬率（未賣出情況）
if not states_sell:
    last_sell_index = len(df_combined) - 1
else:
    last_sell_index = states_sell[-1]


invest = ((original_data[last_sell_index] - original_data[0]) / original_data[0]) * 100

axes1.set_title('COMPLEX  Total Investment: %.2f%%' % invest)
axes1.plot(df_combined.index, signals_combined['short_5Dma'], label='Short MA (5 days)', linestyle='dotted', alpha=0.7)
axes1.plot(df_combined.index, signals_combined['medium_20Dma'], label='Medium MA (20 days)', linestyle='dotted', alpha=0.7)
axes1.plot(df_combined.index, signals_combined['long_60Dma'], label='Long MA (60 days)', linestyle='dotted', alpha=0.7)
axes1.plot(df_combined.index, signals_combined['extra_180Dma'], label='Extra Long MA (180 days)', linestyle='dotted', alpha=0.7)
axes1.legend()



###___MA圖形可視化___###


# 第二张图（放置在第二行的第一列）
axes2 = plt.subplot(gs[1, 0])
axes2.xaxis.set_major_locator(mdates.MonthLocator())
axes2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))


# 股價走勢
axes2.plot(data.index, original_data, label='Original Stock Price', linestyle='dashed', alpha=0.5)

# 實際訓練和預測測試
axes2.plot(data.index[time_steps:len(trainPredict) + time_steps], trainY.flatten(), label='Actual Train', color='blue')
axes2.plot(data.index[len(trainPredict) + time_steps:], testPredict.flatten(), label='Predicted Test', color='orange')



# 繪製未來五天的預測
axes2.plot(future_dates, future_predictions, marker='o', markersize=2, label='Future Prediction', color='green')
# 計算並繪製買賣訊號
states_buy_ma, states_sell_ma, invest_ma = buy_stock_ma(data['Close'], ma_signals['ma_positions'])
axes2.plot(data.index[states_buy_ma], original_data[states_buy_ma], '^', markersize=10, color='m', label='Buy Signal')
axes2.plot(data.index[states_sell_ma], original_data[states_sell_ma], 'v', markersize=10, color='k', label='Sell Signal')

# 計算總收益和投資報酬率（未賣出情況）
if not states_sell_ma:
    last_sell_index = len(data) - 1
else:
    last_sell_index = states_sell_ma[-1]


invest_ma = ((original_data[last_sell_index] - original_data[0]) / original_data[0]) * 100

axes2.set_title('MA  Total Investment: %.2f%%' % invest_ma)
axes2.xaxis.set_major_locator(mdates.MonthLocator())
axes2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
axes2.plot(data.index, ma_signals['short_5Dma'], label='Short MA (5 days)', linestyle='dotted', alpha=0.7)
axes2.plot(data.index, ma_signals['medium_20Dma'], label='Medium MA (20 days)', linestyle='dotted', alpha=0.7)
axes2.plot(data.index, ma_signals['long_60Dma'], label='Long MA (60 days)', linestyle='dotted', alpha=0.7)
axes2.plot(data.index, ma_signals['extra_180Dma'], label='Extra Long MA (180 days)', linestyle='dotted', alpha=0.7)
axes2.legend()



###___MACD圖形可視化___###


# 第三张图（放置在第三行的第一列）
axes3 = plt.subplot(gs[2, 0])
axes3.xaxis.set_major_locator(mdates.MonthLocator())
axes3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# 股價走勢
axes3.plot(data.index, original_data, label='Original Stock Price', linestyle='dashed', alpha=0.5)

# 實際訓練和預測測試
axes3.plot(data.index[time_steps:len(trainPredict) + time_steps], trainY.flatten(), label='Actual Train', color='blue')
axes3.plot(data.index[len(trainPredict) + time_steps:], testPredict.flatten(), label='Predicted Test', color='orange')




# 繪製未來五天的預測
axes3.plot(future_dates, future_predictions, marker='o', markersize=2, label='Future Prediction', color='green')
# 計算並繪製買賣訊號
states_buy_macd, states_sell_macd, invest_macd = buy_stock_macd(data['Close'], macd_signals['macd_positions'])
axes3.plot(data.index[states_buy_macd], original_data[states_buy_macd], '^', markersize=10, color='m', label='Buy Signal')
axes3.plot(data.index[states_sell_macd], original_data[states_sell_macd], 'v', markersize=10, color='k', label='Sell Signal')

# 計算總收益和投資報酬率（未賣出情況）
if not states_sell_macd:
    last_sell_index = len(data) - 1
else:
    last_sell_index = states_sell_macd[-1]

invest_macd = ((original_data[last_sell_index] - original_data[0]) / original_data[0]) * 100

axes3.set_title('MACD_5_days  Total Investment: %.2f%%' % invest_macd)
axes3.plot(data.index, macd_signals['short_5Dma'], label='Short MA (5 days)', linestyle='dotted', alpha=0.7)
axes3.plot(data.index, macd_signals['medium_20Dma'], label='Medium MA (20 days)', linestyle='dotted', alpha=0.7)
axes3.plot(data.index, macd_signals['long_60Dma'], label='Long MA (60 days)', linestyle='dotted', alpha=0.7)
axes3.plot(data.index, macd_signals['extra_180Dma'], label='Extra Long MA (180 days)', linestyle='dotted', alpha=0.7)
axes3.legend()


###___MACD+5日圖形可視化___###


# # 第四张图（放置在第四行的第一列）
# axes4 = plt.subplot(gs[3, 0])
# axes4.xaxis.set_major_locator(mdates.MonthLocator())
# axes4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

# # 股價走勢
# axes4.plot(data.index, original_data, label='Original Stock Price', linestyle='dashed', alpha=0.5)

# # 實際訓練和預測測試
# axes4.plot(data.index[time_steps:len(trainPredict) + time_steps], trainY.flatten(), label='Actual Train', color='blue')
# axes4.plot(data.index[len(trainPredict) + time_steps:], testPredict.flatten(), label='Predicted Test', color='orange')



# # 繪製未來30天的預測
# axes4.plot(future_dates, future_predictions, marker='o', markersize=2, label='Future Prediction', color='green')
# # 計算並繪製買賣訊號
# states_buy_macd_5, states_sell_macd_5, invest_macd_5 = buy_stock_macd_5(df_5.Close, signal_5s['positions_5'])
# axes4.plot(data.index[states_buy_macd_5], original_data[states_buy_macd_5], '^', markersize=10, color='m', label='Buy Signal')
# axes4.plot(data.index[states_sell_macd_5], original_data[states_sell_macd_5], 'v', markersize=10, color='k', label='Sell Signal')

# # 計算總收益和投資報酬率（未賣出情況）
# if not states_sell_macd_5:
#     last_sell_index = len(data) - 1
# else:
#     last_sell_index = states_sell_macd_5[-1]

# invest_macd_5 = ((original_data[last_sell_index] - original_data[0]) / original_data[0]) * 100

# axes4.set_title('MACD_5_days  Total Investment: %.2f%%' % invest_macd_5)
# axes4.plot(data.index, signal_5s['short_5Dma'], label='Short MA (5 days)', linestyle='dotted', alpha=0.7)
# axes4.plot(data.index, signal_5s['medium_20Dma'], label='Medium MA (20 days)', linestyle='dotted', alpha=0.7)
# axes4.plot(data.index, signal_5s['long_60Dma'], label='Long MA (60 days)', linestyle='dotted', alpha=0.7)
# axes4.plot(data.index, signal_5s['extra_180Dma'], label='Extra Long MA (180 days)', linestyle='dotted', alpha=0.7)
# axes4.legend()

# 调整子图布局
plt.subplots_adjust(top=0.99, bottom=0.1, left=0.1, right=0.85, hspace=0.1)

plt.savefig(f'static/my_plot{stock_code}.png')

# 設定資料庫檔案的路徑
db_path = r'C:/Users/user/Desktop/html/my_database.db'

# 連接到資料庫（如果不存在，則會創建一個新的資料庫）
conn = sqlite3.connect(db_path)

# 創建一個游標對象，用於執行 SQL 語句
cursor = conn.cursor()

# 創建一個表格
cursor.execute('''CREATE TABLE IF NOT EXISTS stocks
                (stock_code TEXT PRIMARY KEY, picture TEXT)''')

# 插入一條資料
stock_code = f"{stock_code}"
picture_path = f"C:/Users/user/Desktop/html/static/my_plot{stock_code}.png"
cursor.execute(f"REPLACE INTO stocks VALUES ('{stock_code}', '{picture_path}')")

# 提交交易
conn.commit()

# 關閉游標和連接
cursor.close()
conn.close()

print("資料已成功插入到資料庫中")

if __name__ == '__main__':
    app.run(debug=True)