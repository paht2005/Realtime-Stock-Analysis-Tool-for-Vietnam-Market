import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from vnstock import Vnstock
from vnstock import *
import time
import warnings

# Disable unnecessary warnings | Tắt cảnh báo không cần thiết
warnings.filterwarnings("ignore")

def format_currency(value):
    """Format currency with thousands separators"""
    return "{:,.0f}".format(value).replace(",", ".")

# 1. Fetch and validate data | Lấy và kiểm tra dữ liệu
def get_intraday_data(symbol, max_retries=5):
    for attempt in range(max_retries):
        try:
            stock = Vnstock().stock(symbol=symbol, source='TCBS')
            data = stock.quote.intraday(symbol=symbol, page_size=10000, show_log=False)
            return data
        except Exception as e:
            if attempt == max_retries - 1:
                raise e

# 2. Pre-processing data | Tiền xử lý dữ liệu
def preprocess_data(df):
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    if df['time'].dt.tz is not None:
        df['time'] = df['time'].dt.tz_convert('UTC').dt.tz_localize(None)
    df['time'] = pd.to_datetime(df['time'])

    df['value'] = df['price'] * df['volume']
    df['in_flow'] = np.where(df['match_type'] == 'Buy', df['value'], 0)
    df['out_flow'] = np.where(df['match_type'] == 'Sell', df['value'], 0)
    df.set_index('time', inplace=True)
    return df

# 3. Summarize & compute metrics | Tổng hợp và tính toán chỉ số
def aggregate_data(df):
    resampled = df.resample('min').agg({
        'in_flow': 'sum',
        'out_flow': 'sum',
        'volume': 'sum',
        'match_type': 'count'
    }).rename(columns={'match_type': 'order_count'})

    resampled['net_flow'] = resampled['in_flow'] - resampled['out_flow']
    resampled['cum_net_flow'] = resampled['net_flow'].cumsum()
    resampled['buy_count'] = df[df['match_type'] == 'Buy'].resample('min')['match_type'].count()
    resampled['sell_count'] = df[df['match_type'] == 'Sell'].resample('min')['match_type'].count()
    resampled['cum_buy'] = resampled['buy_count'].cumsum()
    resampled['cum_sell'] = resampled['sell_count'].cumsum()
    resampled['cum_in_flow'] = resampled['in_flow'].cumsum()
    resampled['cum_out_flow'] = resampled['out_flow'].cumsum()

    resampled['avg_buy_volume'] = np.where(resampled['buy_count'] != 0,
                                           df[df['match_type'] == 'Buy'].resample('min')['volume'].sum() / resampled['buy_count'], 0)
    resampled['avg_sell_volume'] = np.where(resampled['sell_count'] != 0,
                                            df[df['match_type'] == 'Sell'].resample('min')['volume'].sum() / resampled['sell_count'], 0)
    resampled['avg_buy_sell_ratio'] = np.where(resampled['avg_sell_volume'] != 0,
                                               resampled['avg_buy_volume'] / resampled['avg_sell_volume'], np.inf)
    return resampled

# 4. Compute summary statistics | Tính toán thống kê tóm tắt
def calculate_summary(df, resampled):
    volatility = df['price'].std()
    imbalance_ratio = np.where(resampled['out_flow'] != 0, resampled['in_flow'] / resampled['out_flow'], 0)
    order_to_volume_ratio = np.where(resampled['volume'] != 0, resampled['order_count'] / resampled['volume'], 0)

    summary = {
        'Tổng dòng tiền vào (VND)': format_currency(resampled['in_flow'].sum()),
        'Tổng dòng tiền ra (VND)': format_currency(resampled['out_flow'].sum()),
        'Dòng tiền ròng (VND)': format_currency(resampled['net_flow'].sum()),
        'Tổng số lệnh mua': int(resampled['buy_count'].sum()),
        'Tổng số lệnh bán': int(resampled['sell_count'].sum()),
        'Khối lượng trung bình lệnh mua': resampled['avg_buy_volume'].mean(),
        'Khối lượng trung bình lệnh bán': resampled['avg_sell_volume'].mean(),
        'Tỷ lệ khối lượng trung bình mua/bán': resampled['avg_buy_sell_ratio'].replace(np.inf, 0).mean(),
        'Giá cao nhất': df['price'].max(),
        'Giá thấp nhất': df['price'].min(),
        'Giá trung bình': df['price'].mean(),
        'Volatility (Độ lệch chuẩn giá)': volatility,
        'Imbalance Ratio (Trung bình)': np.mean(imbalance_ratio),
        'Order-to-Volume Ratio (Trung bình)': np.mean(order_to_volume_ratio)
    }
    return summary

# 5. Generate summary table output | In ra bảng tóm tắt
def print_summary(summary):
    print("\n=== TÓM TẮT PHÂN TÍCH ===")
    for key, value in summary.items():
        if isinstance(value, str):
            print(f"{key}: {value}")
        elif isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")
    print("=========================\n")

#  6. Plot all charts | Vẽ tất cả biểu đồ
def plot_all_charts(df, resampled, symbol):
    plot_cum_net_flow(resampled, symbol)
    plot_avg_buy_sell_ratio(resampled, symbol)
    plot_cum_in_out_flow(resampled, symbol)
    plot_net_flow_heatmap(df, symbol)
    plot_volume_and_orders_distribution(df, resampled, symbol)

    # 6.1. Chart of Cumulative Net Cash Flow | Biểu đồ dòng tiền ròng tích lũy
def plot_cum_net_flow(resampled, symbol):
    plt.figure(figsize=(10, 4))
    plt.plot(resampled.index, resampled['cum_net_flow'], label='Cumulative Net Flow', color='blue')
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.title(f'Dòng tiền ròng tích lũy - {symbol}')
    plt.xlabel('Thời gian')
    plt.ylabel('VNĐ')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_cum_net_flow(resampled, symbol):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(resampled.index, resampled['cum_net_flow'], label='Cumulative Net Flow', color='blue')
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.set_title(f'Dòng tiền ròng tích lũy - {symbol}')
    ax.set_xlabel('Thời gian')
    ax.set_ylabel('VNĐ')
    ax.grid(True)
    plt.tight_layout()
    return fig  # Return the figure for Streamlit to display 
    # 6.2. Average Buy/Sell Volume Ratio | Tỷ lệ Mua/Bán trung bình
def plot_avg_buy_sell_ratio(resampled, symbol):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(resampled.index, resampled['avg_buy_sell_ratio'], label='Buy/Sell Ratio', color='purple')
    ax.set_title(f'Tỷ lệ khối lượng trung bình Mua/Bán - {symbol}')
    ax.set_xlabel('Thời gian')
    ax.set_ylabel('Tỷ lệ')
    ax.grid(True)
    plt.tight_layout()
    return fig
    # 6.3. Cumulative Inflow/Outflow | Dòng tiền vào/ra tích lũy
def plot_cum_in_out_flow(resampled, symbol):
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(resampled.index, resampled['cum_in_flow'], label='Cumulative Inflow', color='green')
    ax.plot(resampled.index, resampled['cum_out_flow'], label='Cumulative Outflow', color='red')
    ax.set_title(f'Dòng tiền vào / ra tích lũy - {symbol}')
    ax.set_xlabel('Thời gian')
    ax.set_ylabel('VNĐ')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig
    # 6.4. Heatmap of Net Cash Flow | Heatmap dòng tiền ròng

def plot_net_flow_heatmap(df, symbol):
    df = df.copy()

    # Ensure index is datetime | Đảm bảo index là datetime
    df.index = pd.to_datetime(df.index)

    # Create 'hour' and 'minute' columns for heatmap | Tạo 2 cột giờ và phút để xây heatmap
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute

    # Calculate net flow for each transaction | Tính net flow mỗi dòng
    df['net_flow'] = df['in_flow'] - df['out_flow']

    # Group by hour and minute to create heatmap data | Group theo giờ và phút để tạo heatmap data
    heatmap_data = df.groupby(['hour', 'minute'])['net_flow'].sum().unstack(fill_value=0)

    # heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(heatmap_data, cmap='RdYlGn', cbar_kws={'label': 'Dòng tiền ròng (VNĐ)'})
    plt.title(f'Heatmap dòng tiền ròng theo phút - {symbol}')
    plt.xlabel('Phút')
    plt.ylabel('Giờ')
    plt.tight_layout()
    return plt.gcf()

    # 6.5. Trading Volume and Order Quantity | Khối lượng và số lượng
def plot_volume_and_orders_distribution(df, resampled, symbol):
    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.bar(resampled.index, resampled['volume'], width=0.0005, color='skyblue', label='Volume')
    ax1.set_ylabel('Khối lượng giao dịch', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax2 = ax1.twinx()
    ax2.plot(resampled.index, resampled['order_count'], color='orange', label='Số lệnh')
    ax2.set_ylabel('Số lệnh', color='orange')
    ax2.tick_params(axis='y', labelcolor='orange')
    plt.title(f'Phân bố khối lượng và số lệnh - {symbol}')
    fig.tight_layout()
    return fig

def analyze_stock(symbol):
    """Phân tích chi tiết mã cổ phiếu"""
    try:
        df = get_intraday_data(symbol)
        if df.empty:
            raise ValueError(f"Dữ liệu trống cho mã {symbol}. Mã có thể không tồn tại hoặc chưa có giao dịch.")
        
        df = preprocess_data(df)
        resampled = aggregate_data(df)
        summary = calculate_summary(df, resampled)
        plot_all_charts(df, resampled, symbol)
        print_summary(summary)
    
    except Exception as e:
        print(f"Lỗi phân tích mã {symbol}: {e}")



