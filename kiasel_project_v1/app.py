import os
import importlib
import datetime as dt
from typing import Optional, List, Dict, Any
import pandas as pd
import streamlit as st
import altair as alt
from dotenv import load_dotenv

load_dotenv()

# =========================================================================================================================
#  코어 모듈(ask_adx_nl) 로딩 
# =========================================================================================================================
@st.cache_resource(show_spinner=False)
def load_core_module():
    core_module = os.getenv("CORE_MODULE", "ask_adx_nl")
    try:
        return importlib.import_module(core_module)
    except Exception as e:
        st.error(f"코어 모듈({core_module})을 불러오지 못했습니다.\n에러: {e}")
        st.stop()

core = load_core_module()

# =========================================================================================================================
#  웹페이지 구성
# =========================================================================================================================
st.set_page_config(
    page_title="Message System Web UI",
    page_icon="📊",
    layout="wide",
)

st.title("📊 메시지 통계 시스템")
st.caption("자연어 질문으로 KQL을 생성/실행하거나, 기준일 비교/예측·월/연 추이를 조회합니다.")

# =========================================================================================================================
#  캐시된 리소스 및 데이터 로딩
# =========================================================================================================================
@st.cache_resource(show_spinner=False)
def get_client():
    """Kusto 클라이언트를 생성하고 캐시합니다."""
    return core.build_kusto_client()

@st.cache_data(show_spinner=False, ttl=3600)  # 1시간 캐시
def get_companies(_client=None) -> List[str]:
    """회사 목록을 로드하고 캐시합니다."""
    if _client is None:
        _client = get_client()
    return core.load_known_companies(_client)

# =========================================================================================================================
#  상수 및 설정
# =========================================================================================================================
MODE_OPTIONS = ["단순 조회", "기준일 비교+예측", "월/연 추이 분석"]
CURRENT_YEAR = dt.date.today().year
DEFAULT_BASE_DATE = dt.date(2025, 8, 28)

# =========================================================================================================================
#  세션 상태 초기화 
# =========================================================================================================================
def initialize_session_state():
    defaults = {
        "question": "2025.05 구글의 사용량 궁금해. 일통계로 보여줘",
        "sel_mode": "단순 조회",
        "sel_company": "자동(질문에서 인식)",
        "sel_year": str(CURRENT_YEAR),
        "sel_month": "전체",
        "sel_base_date": DEFAULT_BASE_DATE
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

initialize_session_state()

# =========================================================================================================================
#  유틸리티 함수들
# =========================================================================================================================
def get_year_options() -> List[str]:
    """최근 5년 연도 옵션을 반환합니다."""
    return ["전체"] + [str(year) for year in range(CURRENT_YEAR, CURRENT_YEAR - 5, -1)]

def get_month_options() -> List[str]:
    """월 옵션을 반환합니다."""
    return ["전체"] + [str(month) for month in range(1, 13)]

def pick_company_from_question(question: str, companies: List[str]) -> Optional[str]:
    """질문에서 회사를 추출하거나 수동 선택된 회사를 반환합니다."""
    if st.session_state["sel_company"] != "자동(질문에서 인식)":
        return st.session_state["sel_company"]
    
    _, used = core.apply_aliases(question, companies)
    return used.get("company_found")

def _build_period_string(year: str, month: str) -> str:
    """연도와 월 정보를 바탕으로 기간 문자열을 생성합니다."""
    if year == "전체":
        return "전체 연도"
    elif month != "전체":
        return f"{year}년 {int(month)}월"
    else:
        return f"{year}년"

def build_question_template() -> str:
    """사이드바 설정을 바탕으로 질문 템플릿을 생성합니다."""
    comp = st.session_state.get("sel_company", "자동(질문에서 인식)")
    year = st.session_state.get("sel_year", str(CURRENT_YEAR))
    month = st.session_state.get("sel_month", "전체")
    mode = st.session_state.get("sel_mode", "단순 조회")
    base_date = st.session_state.get("sel_base_date", dt.date.today())

    parts = []
    
    # 회사명 추가
    if comp and comp != "자동(질문에서 인식)":
        parts.append(str(comp))

    # 모드별 템플릿
    if mode == "기준일 비교+예측":
        base_str = base_date.strftime("%Y-%m-%d") if isinstance(base_date, dt.date) else str(base_date)
        parts.append(f"{base_str} 기준 전일/1주전/1개월전 비교하고 다음 7일 예측도 보여줘!")
    elif mode == "월/연 추이 분석":
        period_str = _build_period_string(year, month)
        parts.append(f"{period_str} 추이 분석")
    else:  # "단순 조회"
        period_str = _build_period_string(year, month)
        parts.append(f"{period_str} 상위 10개")

    question = " ".join(parts).strip()
    return question if question else "삼성 2025-05-05 기준 전일/1주전/1개월전 비교하고 다음 7일 예측도 보여줘"

# =========================================================================================================================
#  차트 관련 유틸리티 함수들
# =========================================================================================================================
class ChartUtils:
    """차트 생성을 위한 유틸리티 클래스"""
    
    @staticmethod
    def find_numeric_column(df: pd.DataFrame, exclude_cols=("year", "hour", "month")) -> Optional[str]:
        """차트 y축으로 사용할 첫 번째 수치 컬럼을 찾습니다."""
        if df is None or df.empty:
            return None
        
        for col in df.columns:
            if col not in exclude_cols and pd.api.types.is_numeric_dtype(df[col]):
                return col
        return None
    
    @staticmethod
    def parse_date_column(df: pd.DataFrame, candidates=("date", "day", "dt", "timestamp", "ts")) -> Optional[str]:
        """날짜 컬럼을 찾아 datetime으로 변환하고 컬럼명을 반환합니다."""
        if df is None or df.empty:
            return None
        
        for col in candidates:
            if col in df.columns:
                try:
                    df[col] = pd.to_datetime(df[col])
                    return col
                except Exception:
                    continue
        return None
    
    @staticmethod
    def get_weekday_order(values: pd.Series) -> List[str]:
        """요일 컬럼의 정렬 순서를 반환합니다."""
        unique_values = values.dropna().unique()
        
        # 숫자 형태인지 확인
        if all(str(v).isdigit() for v in unique_values):
            return [str(i) for i in range(7)]
        
        # 한글 요일
        ko_weekdays = ["월", "화", "수", "목", "금", "토", "일"]
        if any(v in ko_weekdays for v in unique_values):
            return ko_weekdays
        
        # 영문 요일
        en_weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        if any(v in en_weekdays for v in unique_values):
            return en_weekdays
        
        # 기본: 알파벳순
        return sorted(str(v) for v in unique_values)

# =========================================================================================================================
#  UI 렌더링 함수들
# =========================================================================================================================
def render_sidebar():
    """사이드바 UI를 렌더링합니다."""
    companies = get_companies(get_client())
    
    def sync_question():
        st.session_state["question"] = build_question_template()
    
    # 모드 선택
    st.sidebar.radio(
        "모드 선택",
        MODE_OPTIONS,
        key="sel_mode",
        on_change=sync_question,
    )
    
    # 회사 선택
    company_options = ["자동(질문에서 인식)"] + companies
    st.sidebar.selectbox(
        "회사 선택",
        options=company_options,
        key="sel_company",
        on_change=sync_question,
    )
    
    # KQL 표시 옵션
    show_kql = st.sidebar.checkbox("KQL 보기", value=True)

    # === 연도/월 선택 (기준일 비교+예측 모드면 비활성화) ===
    mode = st.session_state.get("sel_mode", "단순 조회")
    disable_period = (mode == "기준일 비교+예측")
    
    # 연도/월 선택
    col_y, col_m = st.sidebar.columns(2)
    with col_y:
        st.selectbox(
            "연도",
            get_year_options(),
            key="sel_year",
            on_change=sync_question,
            disabled=disable_period,        
            help="기준일 비교+예측 모드에서는 연/월 선택을 사용하지 않습니다." if disable_period else None
        )
    with col_m:
        st.selectbox(
            "월",
            get_month_options(),
            key="sel_month",
            on_change=sync_question,
            disabled=disable_period,        # ← 여기!
            help="기준일 비교+예측 모드에서는 연/월 선택을 사용하지 않습니다." if disable_period else None
        )
    
    # 기준일 (기준일 비교+예측 모드에서만 표시)
    if st.session_state["sel_mode"] == "기준일 비교+예측":
        st.sidebar.markdown("---")
        st.sidebar.write("**기준일 선택** (질문에 자동 반영)")
        st.sidebar.date_input(
            "기준일",
            key="sel_base_date",
            on_change=sync_question,
        )
    
    return show_kql, companies

def show_dataframe(label: str, df: pd.DataFrame):
    """데이터프레임을 표시합니다."""
    st.subheader(label)
    if df is None or df.empty:
        st.info("데이터가 없습니다.")
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)

# =========================================================================================================================
#  차트 생성
# =========================================================================================================================
def create_chart(df: pd.DataFrame, chart_type: str, title: str = "") -> Optional[alt.Chart]:
    """차트 타입에 따라 Altair 차트를 생성합니다."""
    if df is None or df.empty:
        return None
    
    df = df.copy()
    chart_utils = ChartUtils()
    
    if chart_type == "daily":
        date_col = chart_utils.parse_date_column(df)
        y_col = chart_utils.find_numeric_column(df)
        if not (date_col and y_col):
            return None
        
        color_enc = alt.Color("year:N", title="연도") if "year" in df.columns else alt.value("steelblue")
        return alt.Chart(df).mark_line(point=True).encode(
            x=alt.X(f"{date_col}:T", title="일자"),
            y=alt.Y(f"{y_col}:Q", title="합계"),
            color=color_enc,
            tooltip=[date_col, y_col] + (["year"] if "year" in df.columns else [])
        ).interactive().properties(title=title)
    
    elif chart_type == "monthly":
        month_col = next((c for c in ("month", "월", "Month", "mon") if c in df.columns), None)
        y_col = chart_utils.find_numeric_column(df)
        if not (month_col and y_col):
            return None

        s = df[month_col]

        # 1) month_num 만들기 (1~12)
        if pd.api.types.is_datetime64_any_dtype(s):
            df["_month_num"] = s.dt.month
        elif pd.api.types.is_numeric_dtype(s):
            # 에포크 값이면 s(초) 또는 ms(밀리초)로 판단해서 변환
            ser = s.dropna()
            if not ser.empty and ser.abs().gt(1000).any():
                med = ser.median()
                unit = "ms" if med > 10**10 else "s"
                df["_month_num"] = pd.to_datetime(s, unit=unit, errors="coerce").dt.month
            else:
                df["_month_num"] = s.astype(int)
        else:
            # 문자열인 경우: 날짜 파싱 시도 → 실패하면 '01월'/'1' 같은 패턴에서 숫자만 추출
            try:
                df["_month_num"] = pd.to_datetime(s, errors="coerce").dt.month
            except Exception:
                df["_month_num"] = (
                    s.astype(str).str.extract(r'(\d{1,2})')[0].astype(float)
                )
            df["_month_num"] = df["_month_num"].fillna(0).astype(int)

        # 2) 레이블/정렬 생성
        df["_month_label"] = df["_month_num"].clip(lower=1, upper=12).astype(int).astype(str) + "월"
        order = [f"{i}월" for i in range(1, 13)]

        color_enc = alt.Color("year:N", title="연도") if "year" in df.columns else alt.value("steelblue")
        return alt.Chart(df).mark_bar().encode(
            x=alt.X("_month_label:O", sort=order, title="월"),
            y=alt.Y(f"{y_col}:Q", title="합계"),
            color=color_enc,
            tooltip=["_month_label", y_col] + (["year"] if "year" in df.columns else [])
        ).properties(title=title)    
    
    elif chart_type == "weekday":
        wd_col = next((c for c in ("weekday", "요일", "dow", "dayofweek") if c in df.columns), None)
        y_col = chart_utils.find_numeric_column(df)
        if not (wd_col and y_col):
            return None
        
        order = chart_utils.get_weekday_order(df[wd_col].astype(str))
        color_enc = alt.Color("year:N", title="연도") if "year" in df.columns else alt.value("steelblue")
        return alt.Chart(df).mark_bar().encode(
            x=alt.X(f"{wd_col}:O", sort=order, title="요일"),
            y=alt.Y(f"{y_col}:Q", title="합계"),
            color=color_enc,
            tooltip=[wd_col, y_col] + (["year"] if "year" in df.columns else [])
        ).properties(title=title)
    
    elif chart_type == "hourly":
        hr_col = next((c for c in ("hour", "시간", "hour_of_day", "hod") if c in df.columns), None)
        y_col = chart_utils.find_numeric_column(df)
        if not (hr_col and y_col):
            return None

        color_enc = alt.Color("year:N", title="연도") if "year" in df.columns else alt.value("steelblue")
        return alt.Chart(df).mark_bar().encode(
            x=alt.X(f"{hr_col}:Q", title="시간(0~23)"),   # <-- 숫자형 축으로!
            y=alt.Y(f"{y_col}:Q", title="합계"),
            color=color_enc,
            tooltip=[hr_col, y_col] + (["year"] if "year" in df.columns else [])
        ).properties(title=title)
    
    return None

# =========================================================================================================================
#  차트 구성
# =========================================================================================================================
def render_trend_analysis_charts(dfs: Dict[str, pd.DataFrame]):
    st.subheader("시각화")
    
    chart_configs = [
        ("일별", "daily", dfs.get("daily")),
        ("월별", "monthly", dfs.get("monthly")),
        ("요일별", "weekday", dfs.get("weekday")),
        ("시간대", "hourly", dfs.get("hour"))
    ]
    
    tabs = st.tabs([config[0] for config in chart_configs])
    
    for i, (tab_name, chart_type, data) in enumerate(chart_configs):
        with tabs[i]:
            if data is None or data.empty:
                st.info(f"{tab_name} 데이터가 없습니다.")
            else:
                chart = create_chart(data, chart_type, f"{tab_name} 추이")
                if chart:
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info(f"{tab_name} 차트를 그릴 수 있는 컬럼 구성이 아닙니다.")

# =========================================================================================================================
#  비즈니스 로직 처리 함수 : 기준일 비교 + 예측 실행
# =========================================================================================================================
def execute_comparison_and_forecast(question: str, companies: List[str], show_kql: bool):
    
    # 회사명 파싱 (필수는 아님)
    company = pick_company_from_question(question, companies)
    if not company:
        company = "" 
   
    # 기준일 파싱 (질문에서 없으면 "기준일 선택"에서 세팅 )
    base_date = core.parse_exact_date(question)
    if not base_date:
        base_date = st.session_state["sel_base_date"].strftime("%Y-%m-%d")
    
    # 전일/1주전 동일요일/1개월전 동일일의 합계를 비교하고 증감률을 계산
    kql_cmp = core.build_point_compare_kql(base_date, company)
    # 기준일 직전 180일 기반으로 7일 예측.
    kql_fc = core.build_next7_forecast_kql(base_date, company)
    
    # KQL 보기 옵션
    if show_kql:
        st.code(kql_cmp, language="kql")
        st.code(kql_fc, language="kql")
    
    # 쿼리 실행
    with st.spinner("ADX 쿼리 실행 중..."):
        df_cmp = core.run_kql(get_client(), kql_cmp)
        df_fc = core.run_kql(get_client(), kql_fc)
    
    # 결과 표시
    show_dataframe("기준일 비교 (전일/1주전/1개월전)", df_cmp)
    show_dataframe("다음 7일 예측", df_fc)
    
    # AI 요약 코멘트
    st.subheader("요약 코멘트")
    summary = core.analyze_point_and_forecast(df_cmp, df_fc, base_date, company)
    st.markdown(summary.replace("\n", "  \n"))

# =========================================================================================================================
# 비즈니스 로직 처리 함수 : 월/연 추이 분석
# =========================================================================================================================
def execute_trend_analysis(question: str, companies: List[str], show_kql: bool):

    # 회사 정보는 필수 
    company = pick_company_from_question(question, companies)
    if not company:
        st.warning("회사명을 인식하지 못했습니다. 왼쪽에서 회사를 수동 선택해주세요.")
        return
    
    # 연도 설정  
    year_str = st.session_state["sel_year"]
    if year_str == "전체":
        years = [int(y) for y in get_year_options()[1:]]  
    else:
        years = [int(year_str)]
        
    # 월 설정
    month_str = st.session_state["sel_month"]
    month = None if month_str == "전체" else int(month_str)
    
    # 데이터 수집
    agg_data = {key: [] for key in ["daily", "monthly", "weekday", "hour", "topdays", "lowdays"]}
    
    for year in years:
        label = f"{year}년" if month is None else f"{year}년 {month}월"
        kqls = core.build_analysis_kqls(year, company, month=month)
        
        if show_kql:
            for key, kql in kqls:
                st.code(f"-- {label} / {key} --\n{kql}", language="kql")
        
        with st.spinner(f"ADX 분석 쿼리 실행 중... ({label})"):
            dfs_year = core.run_kql_all(get_client(), kqls, show_kql=False)
        
        # 연도 정보 추가 및 집계
        for key in agg_data.keys():
            df = dfs_year.get(key)
            if df is not None and not df.empty:
                df = df.copy()
                df["year"] = year
                agg_data[key].append(df)
    
    # 데이터 결합
    dfs = {}
    for key, parts in agg_data.items():
        if len(years) == 1:
            dfs[key] = parts[0] if parts else pd.DataFrame()
        else:
            dfs[key] = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    
    # 차트 렌더링
    render_trend_analysis_charts(dfs)
    
    # 데이터 테이블 표시
    for key, label in [
        ("daily", "일별 합계"), ("monthly", "월별 합계"), 
        ("weekday", "요일별 합계"), ("hour", "시간대 합계"),
        ("topdays", "Top 5 일자"), ("lowdays", "Low 5 일자")
    ]:
        show_dataframe(label, dfs.get(key))
    
    # 요약 코멘트
    st.subheader("요약 코멘트")
    period_label = f"전체({years[0]}~{years[-1]})" if len(years) > 1 else f"{years[0]}년"
    if month:
        period_label += f" {month}월"
    
    try:
        summary = core.analyze_and_comment(dfs, company, period_label)
    except TypeError:
        summary = core.analyze_and_comment(dfs, years[0], company)
    
    st.markdown(summary.replace("\n", "  \n"))

# =========================================================================================================================
# 비즈니스 로직 처리 함수 : 단순 조회
# =========================================================================================================================
def execute_simple_query(question: str, show_kql: bool):
    if show_kql:
        st.info("자연어 → KQL 변환 후 단일 쿼리를 실행합니다.")
    
    with st.spinner("KQL 생성 중..."):
        kql = core.question_to_kql(question)
    
    if show_kql:
        st.code(kql or "<empty>", language="kql")
    
    if not kql:
        st.error("KQL 생성에 실패했습니다. 질문을 바꿔보세요.")
        return
    
    with st.spinner("ADX 실행 중..."):
        df = core.run_kql(get_client(), kql)
    
    show_dataframe("결과", df)

# =========================================================================================================================
#  메인 실행 로직
# =========================================================================================================================
def main():

    # 사이드바 렌더링
    show_kql, companies = render_sidebar()
    
    # 질문 입력
    question = st.text_area("질문 입력", key="question", height=90)
    execute = st.button("실행", type="primary")
    
    if execute:
        try:
            company = pick_company_from_question(question, companies)
            mode = st.session_state["sel_mode"]
            
            st.write(f"**선택된 모드**: `{mode}`")
            st.write(f"**회사 인식 결과**: `{company or '미인식'}`")
            
            # 모드별 실행
            if mode == "기준일 비교+예측":
                execute_comparison_and_forecast(question, companies, show_kql)
            elif mode == "월/연 추이 분석":
                execute_trend_analysis(question, companies, show_kql)
            else:  # "단순 조회"
                execute_simple_query(question, show_kql)
                
        except Exception as e:
            st.error(f"실행 중 오류: {e}")
            if st.checkbox("상세 오류 정보 표시"):
                import traceback
                st.code(traceback.format_exc())

if __name__ == "__main__":
    main()