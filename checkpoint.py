import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vnstock import Vnstock
from vnstock import *
from stock_analyzer import * 
from datetime import datetime, timedelta

# Hàm format tiền tệ
def format_currency(value):
    return f"{value:,.0f} ₫"

# ============ Streamlit UI ============


st.set_page_config(page_title="Phân Tích Cổ Phiếu", layout="wide")
st.title("📈 Hệ Thống Phân Tích Cổ Phiếu")
st.markdown("""
Nhập mã cổ phiếu (ví dụ: **VIC**, **VNM**, **ACB**...) để xem phân tích dữ liệu giao dịch trong ngày.
""")

symbol = st.text_input("Nhập mã cổ phiếu (gõ END để kết thúc):").strip().upper()

if symbol == 'END':
    st.success("Kết thúc phiên làm việc. Tạm biệt!")
    st.stop()
if symbol:
    try:
        st.info(f"Đang tải dữ liệu cho mã **{symbol}**...")

        df = get_intraday_data(symbol)
        if df.empty:
            st.warning(f"Không có dữ liệu cho mã {symbol}.")
        else:
            df = preprocess_data(df)
            resampled = aggregate_data(df)
            summary = calculate_summary(df, resampled)

            # Tabs for charts and summary
            tab1, tab2 = st.tabs(["📊 Biểu đồ", "📋 Thống kê"])

            with tab1:
                st.subheader("Biểu đồ phân tích")

                st.pyplot(plot_cum_net_flow(resampled, symbol))
                st.pyplot(plot_avg_buy_sell_ratio(resampled, symbol))
                st.pyplot(plot_cum_in_out_flow(resampled, symbol))
                st.pyplot(plot_net_flow_heatmap(df, symbol))
                st.pyplot(plot_volume_and_orders_distribution(df, resampled, symbol))

            with tab2:
                st.subheader("Tóm tắt thống kê")
                for key, value in summary.items():
                    if isinstance(value, str):
                        st.markdown(f"**{key}**: {value}")
                    elif isinstance(value, float):
                        st.markdown(f"**{key}**: {value:.6f}")
                    else:
                        st.markdown(f"**{key}**: {value}")

    except Exception as e:
        st.error(f"Lỗi phân tích mã {symbol}: {e}")
