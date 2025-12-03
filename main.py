import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Streamlit 페이지 설정
st.set_page_config(
    page_title="운동 데이터 상관관계 분석",
    layout="wide"
)

def load_data(uploaded_file):
    """업로드된 파일을 읽어 데이터프레임으로 반환합니다."""
    if uploaded_file is not None:
        try:
            # 파일 확장자를 기반으로 읽기
            file_extension = uploaded_file.name.split('.')[-1].lower()
            
            if file_extension == 'csv':
                # 데이터 로드 시 인코딩 문제 방지를 위해 'cp949' 또는 'euc-kr' 시도
                try:
                    # Streamlit FileUploader 객체는 직접 pd.read_csv의 첫 번째 인자로 사용될 수 있습니다.
                    df = pd.read_csv(uploaded_file, encoding='cp949')
                except UnicodeDecodeError:
                    df = pd.read_csv(uploaded_file, encoding='euc-kr')
                return df
            else:
                st.error("지원되지 않는 파일 형식입니다. CSV 파일을 업로드해주세요.")
                return None

        except Exception as e:
            st.error(f"데이터 로드 중 오류 발생: {e}")
            return None
    return None

def analyze_correlation(df, target_col='체지방율'):
    """
    데이터프레임에서 수치형 컬럼을 추출하고, 
    기준 컬럼과의 상관관계를 분석합니다.
    (이 함수는 변경하지 않았습니다.)
    """
    # 숫자형 데이터만 추출
    numeric_df = df.select_dtypes(include=np.number)
    
    # 체지방율 컬럼이 있는지 확인
    if target_col not in numeric_df.columns:
        st.warning(f"데이터에 '{target_col}' 컬럼이 없습니다. 분석을 건너뜁니다.")
        return None, None, None
        
    correlation_series = numeric_df.corr()[target_col].sort_values(ascending=False)
    correlation_series = correlation_series.drop(target_col, errors='ignore')
    top_5_abs_corr = correlation_series.abs().sort_values(ascending=False).head(5).index.tolist()
    full_corr_matrix = numeric_df.corr()
    
    return correlation_series, full_corr_matrix, top_5_abs_corr

# --- 메인 앱 로직 ---
def main():
    st.title("🏃‍♂️ 운동 데이터 상관관계 분석 웹페이지")
    st.markdown("---")

    # 기존 파일 경로 대신 Streamlit File Uploader 사용
    uploaded_file = st.sidebar.file_uploader(
        "CSV 파일 업로드", 
        type=['csv'],
        help="분석할 운동 데이터를 CSV 파일 형식으로 업로드해주세요."
    )

    df = None
    if uploaded_file is not None:
        df = load_data(uploaded_file)
        
    if df is not None:
        st.sidebar.success(f"'{uploaded_file.name}' 데이터 로드 완료.")
        
        # 사용 가능한 수치형 컬럼 목록 추출
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        
        # 2. 분석 설정 및 데이터 분석
        target_column = st.sidebar.selectbox(
            "분석 기준 속성 선택:", 
            options=numeric_cols,
            index=numeric_cols.index('체지방율') if '체지방율' in numeric_cols else 0 # 기본값: 체지방율
        )

        corr_series, full_corr_matrix, top_5_cols = analyze_correlation(df, target_column)

        if corr_series is not None and not corr_series.empty:
            # --- 분석 결과 요약 (생략, 기존 코드와 동일) ---
            st.header(f"📊 '{target_column}'과의 상관관계 분석")
            # ... (이하 동일한 분석 및 시각화 코드) ...
            
            # 가장 상관관계가 높은 속성
            highest_corr_col = corr_series.idxmax()
            highest_corr_val = corr_series.max()
            
            # 가장 상관관계가 낮은 (음의 상관) 속성
            lowest_corr_col = corr_series.idxmin()
            lowest_corr_val = corr_series.min()

            col1, col2 = st.columns(2)
            
            with col1:
                st.info(f"**가장 높은 양의 상관관계:** **`{highest_corr_col}`** (상관계수: `{highest_corr_val:.3f}`)")
            
            with col2:
                st.info(f"**가장 높은 음의 상관관계:** **`{lowest_corr_col}`** (상관계수: `{lowest_corr_val:.3f}`)")

            st.markdown("---")

            # --- 산점도 그래프 ---
            st.subheader(f"📈 '{target_column}'과 상위 5개 속성의 산점도")
            st.markdown(f"**'{target_column}'**과 **절대값 기준**으로 상관관계가 가장 높은 **상위 5개 속성**과의 관계를 산점도로 확인합니다.")
            
            for col in top_5_cols:
                scatter_fig = px.scatter(
                    df, 
                    x=col, 
                    y=target_column, 
                    trendline="ols",
                    title=f"**{target_column}** vs **{col}** (상관계수: {corr_series[col]:.3f})",
                    height=400
                )
                scatter_fig.update_layout(xaxis_title=col, yaxis_title=target_column)
                st.plotly_chart(scatter_fig, use_container_width=True)

            st.markdown("---")
            
            # --- 히트맵 그래프 ---
            st.subheader("🔥 전체 수치형 속성 간 상관관계 히트맵")
            heatmap_fig = px.imshow(
                full_corr_matrix,
                text_auto=".2f", # 소수점 둘째 자리까지 표시
                aspect="auto",
                color_continuous_scale=px.colors.diverging.RdBu,
                color_continuous_midpoint=0,
                title="전체 상관관계 매트릭스 히트맵"
            )
            heatmap_fig.update_layout(height=800)
            st.plotly_chart(heatmap_fig, use_container_width=True)
            
        elif df is not None:
             st.warning("분석에 필요한 수치형 데이터가 충분하지 않거나, 선택된 기준 속성으로 분석을 수행할 수 없습니다.")
    else:
        st.info("시작하려면 왼쪽 사이드바에서 CSV 파일을 업로드해주세요.")

if __name__ == "__main__":
    main()
