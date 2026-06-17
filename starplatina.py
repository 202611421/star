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

# 1. 외부 파일 없이 인터넷 주소(GitHub Raw)에서 데이터를 직접 긁어와 학습
@st.cache_resource
def train_model_from_url():
    # 깃허브에 오픈된 Kaggle Star Dataset의 원본 Raw 주소 연동
    url = "https://github.com"
    
    # on_bad_lines='skip' 옵션을 추가하여 데이터가 깨지는 토큰 에러를 원천 차단
    df = pd.read_csv(url, on_bad_lines='skip')
    df.columns = df.columns.str.strip()
    
    # 데이터셋에 맞게 컬럼명 매칭 규칙 지정
    temp_col = 'Temperature (K)'
    spec_col = 'Spectral Class'
    
    # 🪐 과학적 온도 정화 필터 가동
    def strict_fix(row):
        t = row[temp_col]
        if t >= 30000: return 'O'
        elif t >= 10000: return 'B'
        elif t >= 7500: return 'A'
        elif t >= 6000: return 'F'
        elif t >= 5200: return 'G'
        elif t >= 3900: return 'K'
        else: return 'M'
    df[spec_col] = df.apply(strict_fix, axis=1)
    
    # 색상 2차 정밀 정제 (공백 및 whitish 처리 완벽 반영)
    df['Star color'] = df['Star color'].str.lower().str.replace('-', ' ').str.strip()
    df['Star color'] = df['Star color'].str.replace(r'\s+', ' ', regex=True)
    strict_color_dict = {
        'blue white': 'blue-white', 'yellowish white': 'yellow-white',
        'yellow white': 'yellow-white', 'white yellow': 'yellow-white',
        'whitish': 'white', 'yellowish': 'yellow',
        'orange red': 'orange-red', 'pale yellow orange': 'orange'
    }
    df['Star color'] = df['Star color'].replace(strict_color_dict)
    
    # 훈련셋 분리 및 원-핫 인코딩
    X = df.drop(['Star type', spec_col], axis=1)
    y = df[spec_col]
    X = pd.get_dummies(X, drop_first=True)
    train_columns = X.columns.tolist()
    
    # 내부 즉시 학습 가동 (0.01초 소요)
    gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gb_model.fit(X, y)
    
    return gb_model, train_columns, df, temp_col, spec_col

try:
    model, train_columns, df, temp_col, spec_col = train_model_from_url()
    
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
            'Luminosity (L/Lo)': lum,
            'Radius (R/Ro)': rad,
            'Absolute magnitude (Mv)': mag,
            'Star color': color.lower().replace('-', ' ').strip()
        }])
        
        input_data['Star color'] = input_data['Star color'].replace({
            'blue white': 'blue-white', 'yellowish white': 'yellow-white',
            'yellow white': 'yellow-white', 'white yellow': 'yellow-white',
            'whitish': 'white', 'yellowish': 'yellow',
            'orange red': 'orange-red', 'pale yellow orange': 'orange'
        })
        input_df = pd.get_dummies(input_data, columns=['Star color'])
        
        for col in train_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[train_columns]
        
        # AI 예측 및 결과 시각화
        prediction = model.predict(input_df)
        probabilities = model.predict_proba(input_df)
        max_proba = max(probabilities) * 100
        
        st.markdown("---")
        st.markdown(f"### 🔮 AI 분석 결과")
        st.info(f"이 별은 **{prediction}형** 항성입니다! (확신도: {max_proba:.2f}%)")
        
        prob_df = pd.DataFrame({
            'Spectral Class': model.classes_,
            'Probability (%)': np.round(probabilities * 100, 2)
        })
        fig_prob = px.bar(prob_df, x='Spectral Class', y='Probability (%)', 
                          title="분광형별 매칭 확률 분포", color_discrete_sequence=['#45A29E'])
        fig_prob.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)', font_color='#C5C6C7')
        st.plotly_chart(fig_prob, use_container_width=True)

    with right_col:
        st.subheader("🌌 실시간 인터랙티브 H-R 도표 (배경 데이터 정화 완료)")
        
        # 사용자 입력 별 데이터 행 가공
        user_star = pd.DataFrame([{
            temp_col: temp,
            'Absolute magnitude(Mv)': mag,
            spec_col: f"USER STAR ({prediction}형)",
            'Luminosity (L/Lo)': lum,
            'Radius (R/Ro)': rad,
            'Marker_Size': 30  # 🔥 유저의 야광별 크기는 30으로 지정
        }])
        
        # 배경 데이터셋에 크기 컬럼 추가
        df_copy = df.copy()
        df_copy['Marker_Size'] = 5  # 🔥 배경 별들의 크기는 5로 고정
        
        # 사용자 입력 별과 배경 데이터를 하나로 병합
        plot_df = pd.concat([user_star, df_copy], ignore_index=True)
        spectral_order = [f"USER STAR ({prediction}형)", 'O', 'B', 'A', 'F', 'G', 'K', 'M']
        
        # 포맷팅 에러를 방어한 안전한 Plotly 산점도 빌드
        fig_hr = px.scatter(
            plot_df, x=temp_col, y='Absolute magnitude(Mv)', color=spec_col,
            category_orders={spec_col: spectral_order},
            size='Marker_Size',                      # 🔥 넘파이 배열 대신 데이터프레임 내 수치 컬럼을 지정하여 에러 완벽 해결!
            size_max=25,                             # 최대 마커 스케일 고정
            symbol=spec_col,
            symbol_sequence=['star'] + ['circle']*(len(spectral_order)-1),
            color_discrete_map={f"USER STAR ({prediction}형)": "#66FCF1"}, # 네온 형광 야광색 지정
            title="Hertzsprung-Russell (H-R) Diagram (Real-time Tracking)"
        )
        
        # 천문학 공식 규칙 적용 (X축 역전, Y축 역전)
        fig_hr.update_xaxes(autorange="reverse", title_text="Temperature (K) ← 고온 (왼쪽)")
        fig_hr.update_yaxes(autorange="reverse", title_text="Absolute Magnitude (Mv) ← Bright Star (Top)")
        
        fig_hr.update_layout(
            plot_bgcolor='#0B0C10', paper_bgcolor='#1F2833', font_color='#C5C6C7',
            legend_title_text='항성 분류군', height=650
        )
        
        st.plotly_chart(fig_hr, use_container_width=True)
        st.write("💡 **도표 보는 팁**: 왼쪽 위의 **네온 형광색 별 기호(⭐)**가 현재 슬라이더/숫자로 입력하신 별의 실시간 우주 위치입니다! 수치를 바꾸면 배경 데이터 흐름 위를 실시간으로 움직입니다.")

except Exception as e:
    st.error(f"❌ 시스템 가동 에러: {e}")
