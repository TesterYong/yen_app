import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import feedparser
import plotly.graph_objects as go
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

# 1. 웹페이지 기본 설정
st.set_page_config(page_title="엔화_어디까지_떨어질까", page_icon="💴", layout="wide")
st.title("💴 원/엔(100엔) 환율 AI 예측 대시보드 v2")
st.caption("등락률(%) 기반 시계열 모델 · 미국 10년물 국채금리 및 기술적 지표 반영")

# RSI(상대강도지수) 계산 함수
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# 2. 데이터 수집 및 고도화 모델 학습
@st.cache_data(ttl=3600)
def load_and_predict():
    start_date = '2022-01-01'
    jpy_krw = fdr.DataReader('JPY/KRW', start_date)['Close'] * 100
    usd_krw = fdr.DataReader('USD/KRW', start_date)['Close']
    usd_jpy = fdr.DataReader('USD/JPY', start_date)['Close']
    us10y = fdr.DataReader('US10YT', start_date)['Close']

    df = pd.DataFrame({
        'jpy_krw': jpy_krw,
        'usd_krw': usd_krw,
        'usd_jpy': usd_jpy,
        'us10y': us10y
    }).ffill().dropna()

    # 비율·변화량·기술적 지표 파생변수 생성
    df['ret_1'] = df['jpy_krw'].pct_change(1)
    df['ret_3'] = df['jpy_krw'].pct_change(3)
    df['ret_5'] = df['jpy_krw'].pct_change(5)
    df['ma5_ratio'] = df['jpy_krw'] / df['jpy_krw'].rolling(5).mean() - 1
    df['ma20_ratio'] = df['jpy_krw'] / df['jpy_krw'].rolling(20).mean() - 1
    df['rsi_14'] = compute_rsi(df['jpy_krw'], 14)
    
    # 외부 거시경제 지표의 변화량
    df['usd_krw_ret'] = df['usd_krw'].pct_change(1)
    df['usd_jpy_ret'] = df['usd_jpy'].pct_change(1)
    df['us10y_diff1'] = df['us10y'].diff(1)
    df['us10y_diff5'] = df['us10y'].diff(5)

    # 타겟: 내일 등락률(%)
    df['next_jpy_krw'] = df['jpy_krw'].shift(-1)
    df['target_ret'] = df['jpy_krw'].pct_change(1).shift(-1)

    features = [
        'ret_1', 'ret_3', 'ret_5', 'ma5_ratio', 'ma20_ratio', 'rsi_14',
        'usd_krw_ret', 'usd_jpy_ret', 'us10y_diff1', 'us10y_diff5'
    ]

    valid_df = df.dropna(subset=features + ['target_ret'])
    split = int(len(valid_df) * 0.85)
    train, test = valid_df.iloc[:split].copy(), valid_df.iloc[split:].copy()

    model = LGBMRegressor(
        n_estimators=200,
        learning_rate=0.03,
        max_depth=4,
        subsample_for_bin=500,
        random_state=42,
        verbose=-1
    )
    model.fit(train[features], train['target_ret'])

    test_pred_ret = model.predict(test[features])
    test['pred_jpy_krw'] = test['jpy_krw'] * (1 + test_pred_ret)
    mae = mean_absolute_error(test['next_jpy_krw'], test['pred_jpy_krw'])

    latest_row = df[features].iloc[[-1]]
    tomorrow_ret = model.predict(latest_row)[0]
    current_rate = df['jpy_krw'].iloc[-1]
    tomorrow_pred = current_rate * (1 + tomorrow_ret)
    current_rsi = df['rsi_14'].iloc[-1]

    return df, test, current_rate, tomorrow_pred, tomorrow_ret, mae, current_rsi

with st.spinner("고도화 AI 모델로 시계열 패턴을 분석 중입니다..."):
    df, test, current_rate, tomorrow_pred, tomorrow_ret, mae, current_rsi = load_and_predict()

# 3. 상단 핵심 지표 카드 (KPI)
diff = tomorrow_pred - current_rate
col1, col2, col3, col4 = st.columns(4)
col1.metric("현재 원/엔 환율 (100엔)", f"{current_rate:.2f} 원")
col2.metric("내일 AI 예측 환율", f"{tomorrow_pred:.2f} 원", f"{diff:+.2f} 원 ({tomorrow_ret*100:+.2f}%)")
col3.metric("모델 평균 오차 (MAE)", f"±{mae:.2f} 원")

rsi_status = "과매도 (반등 가능성)" if current_rsi < 30 else ("과매수 (조정 가능성)" if current_rsi > 70 else "중립 구간")
col4.metric("현재 RSI (14일)", f"{current_rsi:.1f}", rsi_status, delta_color="off")

st.divider()

# 4. 차트(Y축 700~910 범위 고정) 및 최신 뉴스 2단 구성
left_col, right_col = st.columns([2, 1])

with left_col:
    st.subheader("🎯 최근 60거래일 실제 환율 vs AI 예측 비교")
    compare_df = pd.DataFrame({
        '실제 환율': test['next_jpy_krw'],
        'AI 예측 환율': test['pred_jpy_krw']
    }).tail(60)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=compare_df.index, y=compare_df['실제 환율'],
        mode='lines', name='실제 환율',
        line=dict(color='#3b82f6', width=2.5)
    ))
    fig.add_trace(go.Scatter(
        x=compare_df.index, y=compare_df['AI 예측 환율'],
        mode='lines', name='AI 예측 환율',
        line=dict(color='#f97316', width=2, dash='dot')
    ))
    fig.update_layout(
        yaxis=dict(range=[700, 910], title="환율 (원)"),  # ← Y축 범위 설정 구간
        xaxis=dict(title="날짜"),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("파란 실선(실제 환율)과 주황 점선(AI 예측)의 움직임이 겹칠수록 예측력이 높습니다.")

with right_col:
    st.subheader("📰 최신 엔화·미국 금리 뉴스")
    rss_url = "https://news.google.com/rss/search?q=엔화+환율+OR+일본은행+OR+미국+국채금리&hl=ko&gl=KR&ceid=KR:ko"
    feed = feedparser.parse(rss_url)
    for entry in feed.entries[:6]:
        st.markdown(f"- [{entry.title}]({entry.link})")
