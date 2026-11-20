import streamlit as st
import FinanceDataReader as fdr
import pandas as pd
import numpy as np
import feedparser
import urllib.parse
import plotly.graph_objects as go
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error

# 1. 웹페이지 기본 설정
st.set_page_config(page_title="엔화 어디까지 떨어질까 · AI 예측기", page_icon="💴", layout="wide")
st.title("💴 원/엔(100엔) 하이브리드 AI 예측 대시보드")
st.caption("시계열 차트 AI + 미국 10년물 국채금리 + 최근 24시간 실시간 경제뉴스 심리 반영")

# RSI(상대강도지수) 계산 함수
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

# 2. 실시간 뉴스 수집 및 금융 감성 분석 함수 (캐시 없이 접속 시마다 실시간 실행)
def analyze_realtime_news():
    queries = [
        "엔화 환율 OR 일본은행 OR BOJ when:1d",
        "미국 국채금리 OR 연준 금리 OR 달러엔 when:1d"
    ]
    
    # 엔화 강세(원/엔 상승) 요인 키워드 및 가중치
    bullish_keywords = {
        "금리 인상": 2.0, "금리인상": 2.0, "매파": 1.5, "긴축": 1.5,
        "구두개입": 2.0, "외환개입": 2.5, "시장 개입": 2.5, "실개입": 2.5,
        "엔화 강세": 2.0, "엔고": 1.8, "엔화 급등": 2.0, "엔화 반등": 1.5,
        "미국 금리 인하": 1.8, "미 금리인하": 1.8, "비둘기": 1.3,
        "증시 급락": 1.5, "나스닥 폭락": 1.5, "안전자산": 1.5, "엔캐리 청산": 2.2,
        "국채금리 하락": 1.3
    }
    
    # 엔화 약세(원/엔 하락) 요인 키워드 및 가중치
    bearish_keywords = {
        "엔저": -1.8, "엔화 약세": -2.0, "엔화 하락": -1.8, "엔화 추락": -2.0,
        "금리 동결": -1.5, "완화 유지": -1.8, "초저금리": -1.5,
        "국채금리 상승": -1.5, "국채금리 급등": -2.0, "미 금리 상승": -1.5,
        "달러 강세": -1.5, "강달러": -1.5, "달러엔 상승": -1.8,
        "물가 상회": -1.3, "인플레 우려": -1.3, "연준 매파": -1.8
    }

    seen_titles = set()
    articles = []
    total_score = 0.0

    for q in queries:
        encoded_q = urllib.parse.quote(q)
        rss_url = f"https://news.google.com/rss/search?q={encoded_q}&hl=ko&gl=KR&ceid=KR:ko"
        feed = feedparser.parse(rss_url)
        
        for entry in feed.entries[:10]:
            title = entry.title
            if title in seen_titles:
                continue
            seen_titles.add(title)

            score = 0.0
            for kw, weight in bullish_keywords.items():
                if kw in title:
                    score += weight
            for kw, weight in bearish_keywords.items():
                if kw in title:
                    score += weight

            total_score += score
            if score > 0:
                tag = "🔺 엔화 강세 요인"
            elif score < 0:
                tag = "🔻 엔화 약세 요인"
            else:
                tag = "➖ 중립/관망"

            articles.append({
                "title": title,
                "link": entry.link,
                "published": entry.get("published", "")[:16],
                "score": score,
                "tag": tag
            })

    # 뉴스 점수를 -100 ~ +100 범위로 정규화
    if len(articles) > 0:
        normalized_score = np.clip((total_score / max(len(articles), 5)) * 40, -100, 100)
    else:
        normalized_score = 0.0

    return articles, normalized_score

# 3. 시계열 데이터 수집 및 차트 AI 모델 학습
@st.cache_data(ttl=1800)  # 30분마다 금융 시세 갱신
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

    df['ret_1'] = df['jpy_krw'].pct_change(1)
    df['ret_3'] = df['jpy_krw'].pct_change(3)
    df['ret_5'] = df['jpy_krw'].pct_change(5)
    df['ma5_ratio'] = df['jpy_krw'] / df['jpy_krw'].rolling(5).mean() - 1
    df['ma20_ratio'] = df['jpy_krw'] / df['jpy_krw'].rolling(20).mean() - 1
    df['rsi_14'] = compute_rsi(df['jpy_krw'], 14)
    
    df['usd_krw_ret'] = df['usd_krw'].pct_change(1)
    df['usd_jpy_ret'] = df['usd_jpy'].pct_change(1)
    df['us10y_diff1'] = df['us10y'].diff(1)
    df['us10y_diff5'] = df['us10y'].diff(5)

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
    chart_ret = model.predict(latest_row)[0]
    current_rate = df['jpy_krw'].iloc[-1]
    current_rsi = df['rsi_14'].iloc[-1]

    return df, test, current_rate, chart_ret, mae, current_rsi

# 실행 및 결과 결합
with st.spinner("실시간 금융 지표와 최근 24시간 뉴스 속보를 분석 중입니다..."):
    df, test, current_rate, chart_ret, mae, current_rsi = load_and_predict()
    articles, news_score = analyze_realtime_news()

# [핵심 결합 로직] 뉴스 심리 점수(-100~+100)를 최대 ±0.35%의 일일 변동 보정치로 환산해 합산
news_ret_impact = (news_score / 100.0) * 0.0035
final_ret = chart_ret + news_ret_impact
tomorrow_pred = current_rate * (1 + final_ret)
diff = tomorrow_pred - current_rate

# 4. 상단 핵심 지표 카드 (KPI)
col1, col2, col3, col4 = st.columns(4)
col1.metric("현재 원/엔 환율 (100엔)", f"{current_rate:.2f} 원")
col2.metric("최종 AI 예측 환율 (내일)", f"{tomorrow_pred:.2f} 원", f"{diff:+.2f} 원 ({final_ret*100:+.2f}%)")

news_label = "엔화 강세 우위" if news_score > 15 else ("엔화 약세 우위" if news_score < -15 else "중립/혼조세")
col3.metric("실시간 뉴스 심리 지수", f"{news_score:+.0f} 점", f"{news_label} ({news_ret_impact*100:+.2f}% 반영)")
col4.metric("차트 모델 평균 오차", f"±{mae:.2f} 원", f"차트 단독 예측: {chart_ret*100:+.2f}%", delta_color="off")

st.divider()

# 5. 차트 및 실시간 뉴스 분석 결과 2단 구성
left_col, right_col = st.columns([1.8, 1.2])

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
        yaxis=dict(range=[700, 910], title="환율 (원)"),
        xaxis=dict(title="날짜"),
        hovermode="x unified",
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"현재 RSI(14일): {current_rsi:.1f} · 파란 실선(실제)과 주황 점선(예측)이 겹칠수록 정확도가 높습니다.")

with right_col:
    st.subheader("📰 최근 24시간 실시간 뉴스 분석")
    if not articles:
        st.info("최근 24시간 내 수집된 주요 엔화/금리 속보가 없습니다.")
    else:
        for art in articles[:8]:
            st.markdown(f"**{art['tag']}** · [{art['title']}]({art['link']})")
