import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Nhập từ vnstock và stock_analyzer
from vnstock import *
from stock_analyzer import *

# ===== CSS TUỲ BIẾN =====
def load_css():
    st.markdown("""
        <style>
            h1, h2, h3 {
                font-family: 'Segoe UI', sans-serif;
                color: #1A237E;
            }
            .card {
                border: 1px solid #ccc;
                border-radius: 10px;
                padding: 15px;
                margin: 10px 0;
                background-color: #FAFAFA;
            }
        </style>
    """, unsafe_allow_html=True)

# ===== HEADER =====
def render_header():
    st.markdown("""
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
            <h1>📊 Phân tích cổ phiếu Việt Nam</h1>
            <p style="color: #424242;">Hệ thống hỗ trợ phân tích kỹ thuật và dòng tiền</p>
        </div>
        <div style="border: 2px solid #1A237E; border-radius: 10px; padding: 5px;">
            <img src="https://cdn-icons-png.flaticon.com/512/8950/8950837.png" width="60">
        </div>
    </div>
    """, unsafe_allow_html=True)

# ===== SIDEBAR =====
def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Tuỳ chọn")
        symbol = st.text_input("Nhập mã cổ phiếu (gõ END để thoát):").strip().upper()
        date_range = st.date_input("Khoảng thời gian", [datetime.now() - timedelta(days=7), datetime.now()])
        st.markdown("---")
        st.info("📬 Liên hệ hỗ trợ: congphatnguyen.work@gmail.com")
        return symbol, date_range

# ===== THỐNG KÊ METRICS =====
def render_metrics(summary):
    st.markdown("""<div class="card"><h4>📌 Tóm tắt nhanh</h4></div>""", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("💸 Dòng tiền vào (VND)", summary['Tổng dòng tiền vào (VND)'])
    col2.metric("💰 Dòng tiền ra (VND)", summary['Tổng dòng tiền ra (VND)'])
    col3.metric("📈 Dòng tiền ròng (VND)", summary['Dòng tiền ròng (VND)'])

# ===== NOTES =====
def render_credit():
    st.markdown("---")
    st.caption("🚀 Phát triển bởi Nguyễn Công Phát | Dữ liệu từ VnStock API")

# ===== MAIN APP =====
def main():
    st.set_page_config(page_title="Phân Tích Cổ Phiếu", layout="wide")
    load_css()
    render_header()
    symbol, date_range = render_sidebar()

    if symbol == "END":
        st.success("Kết thúc phiên làm việc. Tạm biệt!")
        return

    if symbol:
        try:
            st.markdown(f"### 🏷️ Mã phân tích: `{symbol}` | Từ {date_range[0]} đến {date_range[1]}")
            st.info(f"🔄 Đang tải dữ liệu cho mã **{symbol}**...")

            # === Gọi hàm lấy và xử lý dữ liệu ===
            df = get_intraday_data(symbol)
            if df.empty:
                st.warning(f"⚠️ Không có dữ liệu cho mã {symbol}.")
                return

            df = preprocess_data(df)
            resampled = aggregate_data(df)
            summary = calculate_summary(df, resampled)

            # === Hiển thị số liệu tổng hợp ===
            render_metrics(summary)

            # === Tabs hiển thị ===
            tab1, tab2 = st.tabs(["📊 Biểu đồ", "📋 Chi tiết thống kê"])

            with tab1:
                st.subheader("Biểu đồ phân tích")
                st.pyplot(plot_cum_net_flow(resampled, symbol))
                st.pyplot(plot_avg_buy_sell_ratio(resampled, symbol))
                st.pyplot(plot_cum_in_out_flow(resampled, symbol))
                st.pyplot(plot_net_flow_heatmap(df, symbol))
                st.pyplot(plot_volume_and_orders_distribution(df, resampled, symbol))

            with tab2:
                st.subheader("Thống kê chi tiết")
                for key, value in summary.items():
                    if isinstance(value, str):
                        st.markdown(f"**{key}**: {value}")
                    elif isinstance(value, float):
                        st.markdown(f"**{key}**: {value:,.2f}")
                    else:
                        st.markdown(f"**{key}**: {value}")

            render_credit()

        except Exception as e:
            st.error(f"❌ Lỗi khi phân tích mã {symbol}: {e}")

if __name__ == "__main__":
    main()
