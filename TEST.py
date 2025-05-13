import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vnstock import Vnstock
from vnstock import *
from stock_analyzer import * 


symbol = input("\nNhập mã cổ phiếu để xem (gõ END để kết thúc): ").strip().upper()
df = get_intraday_data(symbol)
df = preprocess_data(df)
resampled = aggregate_data(df)
summary = calculate_summary(df, resampled)

print (df.head())