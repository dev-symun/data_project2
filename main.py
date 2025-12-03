import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# Streamlit 페이지 설정
st.set_page_config(
    page_title="운동 데이터 상관관계 분석",
    layout="wide"
)

def load_data(file_path):
    """CSV 파일을 불러와 데이터프레임으로 반환합니다."""
    try:
        # 데이터 로드 시 인코딩 문제 방지를 위해 'cp949' 또는 'euc-kr' 시도
        df = pd.read_csv(file_path, encoding='cp949')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='euc-kr')
        except Exception as e:
            st.error(f"데이터 로드 중 오류 발생: {e}")
            return None
    return df

def analyze_correlation(df, target_col='체지방율'):
    """
    데이터프레임에서 수치형 컬럼을 추출하고, 
    '체지방율'과의 상관관계를 분석합니다.
    """
    # 숫자형 데이터만 추출
    numeric_df = df.select_dtypes(include=np.number)
    
    # 체지방율 컬럼이 있는지 확인
    if target_col not in numeric_df.columns:
        st.warning(f"데이터에 '{target_col}' 컬럼이 없습니다. 분석을 건너뜁니다.")
        return None, None
        
    # '체지방율'과 다른 모든 수치형 컬럼 간의 상관관계 계산
    # .corr() 결과는 Series 형태로 반환됨
    correlation_series = numeric_df.corr()[target_col].sort_values(ascending=False)
    
    # 자기 자신(체지방율)은 제외
    correlation_series = correlation_series.drop(target_col, errors='ignore')
    
    # 절대값 기준으로 상위 5개 속성 추출 (양/음의 상관관계 모두 포함)
    top_5_abs_corr = correlation_series.abs().sort_values(ascending=False).head(5).index.tolist()
    
    # 전체 상관관계 매트릭스 계산
    full_corr_matrix = numeric_df.corr()
    
    return correlation_series, full_corr_matrix, top_5_abs_corr

# --- 메인 앱 로직 ---
def main():
    st.title("🏃‍♂️ 운동 데이터 상관관계 분석 웹페이지")
    st.markdown("---")

    # 파일 이름 설정 (사용자가 업로드한 파일 이름 사용)
    file_path = "fitness data.xlsx - KS_NFA_FTNESS_MESURE_ITEM_MESUR.csv"
    
    st.sidebar.header("⚙️ 분석 설정")
    target_column = st.sidebar.selectbox(
        "분석 기준 속성 선택:", 
        options=['체지방율', '신장', '체중', 'BMI', '절대악력'], # 자주 사용될 만한 컬럼 예시
        index=0 # 기본값: 체지방율
    )

    # 1. 데이터 로드
    df = load_data(file_path)

    if df is not None:
        st.sidebar.success(f"'{file_path}' 데이터 로드 완료.")
        
        # 2. 데이터 분석
        corr_series, full_corr_matrix, top_5_cols = analyze_correlation(df, target_column)

        if corr_series is not None:
            
            # --- 분석 결과 요약 ---
            st.header(f"📊 '{target_column}'과의 상관관계 분석")

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
            
            # 5개의 컬럼에 대해 산점도 그리기
            for i, col in enumerate(top_5_cols):
                scatter_fig = px.scatter(
                    df, 
                    x=col, 
                    y=target_column, 
                    trendline="ols", # 최소 제곱법(OLS) 추세선 추가
                    title=f"**{target_column}** vs **{col}** (상관계수: {corr_series[col]:.3f})",
                    height=400
                )
                scatter_fig.update_layout(
                    xaxis_title=col,
                    yaxis_title=target_column
                )
                st.plotly_chart(scatter_fig, use_container_width=True)

            st.markdown("---")
            
            # --- 히트맵 그래프 ---
            st.subheader("🔥 전체 수치형 속성 간 상관관계 히트맵")
            st.markdown("모든 수치형 데이터 속성 간의 상관관계를 히트맵으로 한눈에 파악할 수 있습니다.")

            # 히트맵 그리기
            heatmap_fig = px.imshow(
                full_corr_matrix,
                text_auto=True,
                aspect="auto",
                color_continuous_scale=px.colors.diverging.RdBu, # 빨강-파랑 계열
                color_continuous_midpoint=0, # 0을 기준으로 색상 중심
                title="전체 상관관계 매트릭스 히트맵"
            )
            heatmap_fig.update_layout(height=800)
            st.plotly_chart(heatmap_fig, use_container_width=True)

        else:
            st.warning(f"선택된 기준 속성 '{target_column}'으로 분석을 수행할 수 없습니다. 데이터와 컬럼명을 확인해 주세요.")

if __name__ == "__main__":
    main()
