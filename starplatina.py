import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import GradientBoostingClassifier

# 0. 페이지 기본 설정 및 우주 스타일 사이버 다크 테마 주입
st.set_page_config(
    page_title="Star Platina - Universe Dashboard", 
    page_icon="🌌", 
    layout="wide"
)

# 사이버 펑크 감성의 커스텀 CSS 주입 (칠흑 같은 우주 테마)
st.markdown("""
    <style>
    .main { background-color: #0B0C10; color: #C5C6C7; }
    h1, h2, h3 { color: #66FCF1 !important; font-family: 'Courier New', monospace; }
    .stButton>button {
        background-color: #45A29E; color: #1F2833; font-weight: bold;
        border: 2px solid #66FCF1; box-shadow: 0 0 10px #66FCF1;
    }
    .stButton>button:hover { background-color: #66FCF1; color: #0B0C10; }
    </style>
    """, unsafe_allow_html=True)

st.title("🌌 STAR PLATINA : 항성 분석 및 H-R 도표 시스템")
st.write("우주 공간의 항성 데이터를 실시간 분석하고 헤르츠스프룽-러셀(H-R) 도표 위의 위치를 시각화합니다.")
st.markdown("---")

# 1. 외부 파일/링크 스트레스 없이 코드 내부 자체 데이터로 즉석 학습 가동
@st.cache_resource
def train_model_safely():
    raw_data = [
        [3068, 0.0024, 0.17, 16.12, 'Red', 'M'], [3453, 0.0006, 0.11, 16.89, 'Red', 'M'],
        [2983, 0.0004, 0.09, 17.55, 'Red', 'M'], [2900, 0.0003, 0.08, 18.23, 'Red', 'M'],
        [5778, 1.0000, 1.00, 4.83, 'Yellow', 'G'], [6000, 1.4000, 1.20, 4.10, 'Yellow White', 'F'],
        [7500, 5.0000, 1.50, 2.50, 'White', 'A'], [9500, 25.0000, 2.00, 1.20, 'White', 'A'],
        [12000, 120.00, 4.00, -1.50, 'Blue White', 'B'], [25000, 15000.0, 8.00, -4.50, 'Blue', 'B'],
        [35000, 100000.0, 12.0, -6.00, 'Blue', 'O'], [40000, 200000.0, 15.0, -7.20, 'Blue', 'O'],
        [3200, 0.001, 0.14, 15.2, 'Red', 'M'], [3600, 0.003, 0.22, 14.1, 'Red', 'M'],
        [8000, 20.0, 1.8, 1.6, 'White', 'A'], [11000, 80.0, 3.1, -0.2, 'Blue White', 'B'],
        [15000, 500.0, 4.5, -2.1, 'Blue White', 'B'], [22000, 5000.0, 6.2, -3.8, 'Blue', 'B'],
        [32000, 50000.0, 10.5, -5.5, 'Blue', 'O'], [38000, 150000.0, 14.1, -6.8, 'Blue', 'O'],
        [3300, 2000.0, 45.0, -3.5, 'Red', 'M'], [3500, 5000.0, 120.0, -4.8, 'Red', 'M'],
        [4000, 10000.0, 250.0, -5.2, 'Orange', 'K'], [4500, 15000.0, 380.0, -5.8, 'Orange', 'K']
    ]
    
    df = pd.DataFrame(raw_data, columns=['Temperature', 'Luminosity', 'Radius', 'Absolute_Magnitude', 'Star_Color', 'Spectral_Class'])
    
    temp_col = 'Temperature'
    lum_col = 'Luminosity'
    rad_col = 'Radius'
    mag_col = 'Absolute_Magnitude'
    color_col = 'Star_Color'
    spec_col = 'Spectral_Class'
    
    df[color_col] = df[color_col].str.lower().str.replace('-', ' ').str.strip()
    
    X = df.drop([spec_col], axis=1)
    y = df[spec_col]
    X = pd.get_dummies(X, drop_first=True)
    train_columns = X.columns.tolist()
    
    gb_model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
    gb_model.fit(X, y)
    
    return gb_model, train_columns, df, temp_col, lum_col, rad_col, mag_col, color_col, spec_col

try:
    model, train_columns, df, temp_col, lum_col, rad_col, mag_col, color_col, spec_col = train_model_safely()
    
    # 레이아웃 분할
    left_col, right_col = st.columns([1, 1.3])
    
    with left_col:
        st.subheader("📝 항성 정보 실시간 스캔")
        temp = st.number_input("🤖 표면 온도 (Temperature in K)", min_value=100, max_value=100000, value=5778, step=100)
        lum = st.number_input("✨ 상대 광도 (Luminosity L/Lo)", min_value=0.0, max_value=1000000.0, value=1.0, step=0.1)
        rad = st.number_input("🪐 상대 반지름 (Radius R/Ro)", min_value=0.0, max_value=100000.0, value=1.0, step=0.1)
        mag = st.number_input("🔅 절대 등급 (Absolute Magnitude Mv)", min_value=-20.0, max_value=30.0, value=4.83, step=0.01)
        
        color_options = ['Yellow', 'Red', 'Blue White', 'White', 'Yellow White', 'Orange', 'Blue', 'Orange Red']
        color = st.selectbox("🎨 항성 색상 선택 (Star Color)", color_options)
        
        # 실시간 사용자 데이터 가공
        input_data = pd.DataFrame([{
            temp_col: temp,
            lum_col: lum,
            rad_col: rad,
            mag_col: mag,
            color_col: color.lower().replace('-', ' ').strip()
        }])
        
        input_df = pd.get_dummies(input_data, columns=[color_col])
        
        for col in train_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[train_columns]
        
        prediction = model.predict(input_df)
        probabilities = model.predict_proba(input_df)
        
        # 🔥 [치명적 에러 해결 완료] 넘파이 배열에서 순수 파이썬 float 숫자로 강제 래핑하여 포맷팅 오류 차단! [6]
        max_proba = float(max(probabilities[0])) * 100
        
        st.markdown("---")
        st.markdown(f"### 🔮 AI 분석 결과")
        st.info(f"이 별은 **{prediction[0]}형** 항성입니다! (확신도: {max_proba:.2f}%)")
        
        # 확률 테이블 가공 및 순수 숫자 타입 고정
        prob_values = [float(p) * 100 for p in probabilities[0]]
        prob_df = pd.DataFrame({
            'Spectral Class': model.classes_,
            'Probability (%)': np.round(prob_values, 2)
        })
        fig_prob = px.bar(prob_df, x='Spectral Class', y='Probability (%)', 
                          title="분광형별 매칭 확률 분포", color_discrete_sequence=['#45A29E'])
        fig_prob.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#C5C6C7')
        st.plotly_chart(fig_prob, use_container_width=True)

    with right_col:
        st.subheader("🌌 실시간 인터랙티브 H-R 도표 (안전 모드 가동)")
        
        user_label = f"USER STAR ({prediction[0]}형)"
        
        user_star = pd.DataFrame([{
            temp_col: temp,
            mag_col: mag,
            spec_col: user_label,
            lum_col: lum,
            rad_col: rad
        }])
        
        plot_df = pd.concat([user_star, df], ignore_index=True)
        spectral_order = [user_label, 'O', 'B', 'A', 'F', 'G', 'K', 'M']
        
        fig_hr = px.scatter(
            plot_df, x=temp_col, y=mag_col, color=spec_col,
            category_orders={spec_col: spectral_order},
            symbol=spec_col,
            symbol_sequence=['star'] + ['circle']*(len(spectral_order)-1),
            color_discrete_map={user_label: "#66FCF1"}, 
            title="Hertzsprung-Russell (H-R) Diagram (Real-time Tracking)"
        )
        
        # 마커 스타일링 안전 지정
        fig_hr.update_traces(marker=dict(size=10)) 
        fig_hr.update_traces(selector=dict(name=user_label), marker=dict(size=22)) 
        
        # 천문학 공식 규칙 적용 (X축 역전, Y축 역전)
        fig_hr.update_xaxes(autorange="reverse", title_text="Temperature (K) ← 고온 (왼쪽)")
        fig_hr.update_yaxes(autorange="reverse", title_text="Absolute Magnitude (Mv) ← Bright Star (Top)")
        
        fig_hr.update_layout(
            plot_bgcolor='#0B0C10', paper_bgcolor='#1F2833', font_color='#C5C6C7',
            legend_title_text='항성 분류군', height=650
        )
        
        st.plotly_chart(fig_hr, use_container_width=True)
        st.write("💡 **도표 보는 팁**: 왼쪽 위의 **네온 형광색 별 기호(⭐)**가 현재 수치 입력창으로 입력하신 별의 실시간 우주 위치입니다! 수치를 바꾸면 배경 데이터 흐름 위를 실시간으로 움직입니다.")

except Exception as e:
    st.error(f"❌ 시스템 가동 에러: {e}")
