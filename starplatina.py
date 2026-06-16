import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px

# 0. 페이지 기본 설정 및 우주 스타일 사이버 다크 테마 주입
st.set_page_config(
    page_title="Star Platina - Universe Dashboard", 
    page_icon="🌌", 
    layout="wide"
)

# 사이버 펑크 감성의 커스텀 CSS 주입 (칠흑 같은 우주 테마)
st.markdown("""
    <style>
    .main {
        background-color: #0B0C10;
        color: #C5C6C7;
    }
    h1, h2, h3 {
        color: #66FCF1 !important;
        font-family: 'Courier New', monospace;
    }
    .stButton>button {
        background-color: #45A29E;
        color: #1F2833;
        font-weight: bold;
        border: 2px solid #66FCF1;
        box-shadow: 0 0 10px #66FCF1;
    }
    .stButton>button:hover {
        background-color: #66FCF1;
        color: #0B0C10;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🌌 STAR PLATINA : 항성 분석 및 H-R 도표 시스템")
st.write("우주 공간의 항성 데이터를 실시간 분석하고 헤르츠스프룽-러셀(H-R) 도표 위의 위치를 시각화합니다.")
st.markdown("---")

# 1. 머신러닝 피클 파일 로드
@st.cache_resource
def load_ml_components():
    with open('star_gb_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('train_columns.pkl', 'rb') as f:
        columns = pickle.load(f)
    return model, columns

try:
    model, train_columns = load_ml_components()
    
    # 레이아웃을 왼쪽(입력창)과 오른쪽(H-R 도표 및 결과) 2분할
    left_col, right_col = st.columns([1, 1.3])
    
    with left_col:
        st.subheader("📝 항성 정보 실시간 스캔")
        temp = st.number_input("🤖 표면 온도 (Temperature in K)", min_value=100, max_value=100000, value=5778, step=100)
        lum = st.number_input("✨ 상대 광도 (Luminosity L/Lo)", min_value=0.0, max_value=1000000.0, value=1.0, step=0.1)
        rad = st.number_input("🪐 상대 반지름 (Radius R/Ro)", min_value=0.0, max_value=100000.0, value=1.0, step=0.1)
        mag = st.number_input("🔅 절대 등급 (Absolute Magnitude Mv)", min_value=-20.0, max_value=30.0, value=4.83, step=0.01)
        
        color_options = ['Yellow', 'Red', 'Blue White', 'White', 'Yellow White', 'Orange', 'Blue', 'Orange Red']
        color = st.selectbox("🎨 항성 색상 선택 (Star Color)", color_options)
        
        # --- 🔮 머신러닝 예측 전처리 프로세스 ---
        input_data = pd.DataFrame([{
            'Temperature (K)': temp,
            'Luminosity(L/Lo)': lum,
            'Radius(R/Ro)': rad,
            'Absolute magnitude(Mv)': mag,
            'Star color': color.lower().replace('-', ' ').strip()
        }])
        
        strict_color_dict = {
            'blue white': 'blue-white', 'yellowish white': 'yellow-white',
            'yellow white': 'yellow-white', 'white yellow': 'yellow-white',
            'whitish': 'white', 'yellowish': 'yellow',
            'orange red': 'orange-red', 'pale yellow orange': 'orange'
        }
        input_data['Star color'] = input_data['Star color'].replace(strict_color_dict)
        input_df = pd.get_dummies(input_data, columns=['Star color'])
        
        for col in train_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[train_columns]
        
        # 예측 및 확률 도출
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        max_proba = max(probabilities) * 100
        
        st.markdown("---")
        st.markdown(f"### 🔮 AI 분석 결과")
        st.info(f"이 별은 **{prediction}형** 항성입니다! (확신도: {max_proba:.2f}%)")
        
        # 확률 막대그래프 분포
        prob_df = pd.DataFrame({
            'Spectral Class': model.classes_,
            'Probability (%)': np.round(probabilities * 100, 2)
        })
        fig_prob = px.bar(prob_df, x='Spectral Class', y='Probability (%)', 
                          title="분광형별 매칭 확률 분포",
                          color_discrete_sequence=['#45A29E'])
        fig_prob.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#C5C6C7')
        st.plotly_chart(fig_prob, use_container_width=True)

    with right_col:
        st.subheader("🌌 실시간 추적 헤르츠스프룽-러셀(H-R) 도표")
        
        # 🚨 대안 프로세스: 원본 파일 없이 유저의 데이터만으로 단독 H-R 맵을 생성
        # 대신 유저가 좌표를 체감할 수 있게 그래프의 전체 캔버스 범위(우주 공간 축)를 넓게 고정합니다.
        user_star_df = pd.DataFrame([{
            'Temperature (K)': temp,
            'Absolute Magnitude (Mv)': mag,
            'Type': f"MY STAR ({prediction}형)"
        }])
        
        # 동적 피드백이 가능한 실시간 인터랙티브 산점도 구현
        fig_hr = px.scatter(
            user_star_df, 
            x='Temperature (K)', 
            y='Absolute Magnitude (Mv)',
            color='Type',
            color_discrete_map={f"MY STAR ({prediction}형)": "#66FCF1"},
            size=[30], # 내 별이 돋보이도록 크기 확장
            symbol_sequence=['star'], # 별 모양 마커 적용
            title="H-R Diagram Space Tracker"
        )
        
        # 천문학 핵심 법칙 적용: X축(온도) 역전, Y축(절대등급) 역전
        fig_hr.update_xaxes(
            autorange="reverse", 
            range=[40000, 2000],  # 우주 공간의 온도 축 범위를 2,000K ~ 40,000K로 고정
            title_text="Temperature (K) ← High Temperature (Left)"
        )
        fig_hr.update_yaxes(
            autorange="reverse", 
            range=[20, -10],     # 우주 공간의 절대등급 범위를 -10 ~ +20등급으로 고정
            title_text="Absolute Magnitude (Mv) ← Bright Star (Top)"
        )
        
        # 우주 공간 다크 테마 스타일 커스텀
        fig_hr.update_layout(
            plot_bgcolor='#0B0C10',
            paper_bgcolor='#1F2833',
            font_color='#C5C6C7',
            height=600
        )
        
        st.plotly_chart(fig_hr, use_container_width=True)
        st.write("💡 **우주 추적 레이더 설명**: 무거운 데이터 파일 없이 가볍게 작동하는 H-R 맵입니다. 왼쪽 입력창에서 수치를 바꾸면 **형광색 야광별(⭐)**이 광활한 우주 격자 공간 위를 실시간으로 워프하며 움직입니다!")

except FileNotFoundError as e:
    st.error(f"❌ 필수 파일(`star_gb_model.pkl` 또는 `train_columns.pkl`)이 깃허브 저장소에 없습니다. 파일명을 확인해 주세요.")
except Exception as e:
    st.error(f"❌ 시스템 가동 에러: {e}")
