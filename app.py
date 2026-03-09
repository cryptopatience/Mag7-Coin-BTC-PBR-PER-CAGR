import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import openai
import google.generativeai as genai
import time

warnings.filterwarnings('ignore')


# OpenAI와 Gemini 임포트 (선택적)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False



# 페이지 설정
st.set_page_config(
    page_title="MAG 9 AI 종합 분석",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================== 로그인 시스템 (수정됨) ====================
def check_password():
    """비밀번호 확인 및 로그인 상태 관리"""
    if st.session_state.get('password_correct', False):
        return True
    
    st.title("🔒 MAG 9 (MAG 7 + COIN + BTC) 종합 분석")
    st.markdown("### AI 기반 투자 분석 시스템")
    
    with st.form("credentials"):
        password = st.text_input("비밀번호 (Password)", type="password", key="input_password")
        submit_btn = st.form_submit_button("로그인", type="primary")

    if submit_btn:
        if "password" not in st.secrets:
            st.error("⚠️ .streamlit/secrets.toml 파일에 password 설정이 없습니다.")
            return False

        if password == st.secrets["password"]:
            st.session_state['password_correct'] = True
            st.rerun()
        else:
            st.error("😕 비밀번호가 올바르지 않습니다.")
    
    return False


if not check_password():
    st.stop()

# ==================== AI 설정 ====================
# Gemini 초기화
GEMINI_ENABLED = False
try:
    if "GEMINI_API_KEY" in st.secrets:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        GEMINI_ENABLED = True
except Exception as e:
    st.warning(f"⚠️ Gemini AI 초기화 실패: {e}")

# OpenAI 초기화
OPENAI_ENABLED = False
OPENAI_CLIENT = None

try:
    if "OPENAI_API_KEY" in st.secrets:
        OPENAI_CLIENT = openai.OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
        OPENAI_ENABLED = True
except Exception as e:
    st.warning(f"⚠️ OpenAI 초기화 실패: {e}")

# 모델 설정
OPENAI_MODEL_MARKET = st.secrets.get("OPENAI_MODEL_MARKET", "gpt-4o-mini")
OPENAI_MODEL_STOCK = st.secrets.get("OPENAI_MODEL_STOCK", "gpt-4o-mini")
OPENAI_MODEL_CHAT = st.secrets.get("OPENAI_MODEL_CHAT", "gpt-4o-mini")

# ==================== 사이드바 ====================
with st.sidebar:
    st.success("✅ 환영합니다!")
    
    if st.button("🚪 로그아웃", use_container_width=True):
        st.session_state['password_correct'] = False
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📊 분석 옵션")
    
    show_technical = st.checkbox("기술적 분석", value=True)
    show_fundamental = st.checkbox("펀더멘털 분석", value=True)
    show_growth = st.checkbox("5개년 성장률 분석", value=True)
    show_ai = st.checkbox("AI Deep Dive 분석", value=True)
    
    st.markdown("---")
    st.markdown("### 🎯 AI 분석 설정")
    
    if show_ai:
        ai_engine = st.selectbox(
            "AI 엔진 선택",
            ["OpenAI GPT-4", "Google Gemini", "Both"],
            index=2
        )
        
        top_n_analysis = st.slider(
            "AI 분석할 상위 종목 수",
            min_value=1,
            max_value=9,
            value=3
        )
    
    st.markdown("---")
    st.markdown("### ℹ️ 정보")
    st.info("""
    **MAG 9 구성:**
    - 📈 MAG 7: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA
    - 💰 COIN: Coinbase
    - ₿ BTC: Bitcoin
    """)

# ==================== MAG 9 자산 정의 ====================
MAG9_ASSETS = {
    'AAPL': {
        'name': 'Apple Inc.',
        'description': '아이폰, 생태계, 온디바이스 AI',
        'sector': 'Technology',
        'industry': 'Consumer Electronics',
        'type': 'Stock'
    },
    'MSFT': {
        'name': 'Microsoft Corporation',
        'description': '클라우드(Azure), 생성형 AI (OpenAI 대주주)',
        'sector': 'Technology',
        'industry': 'Software',
        'type': 'Stock'
    },
    'GOOGL': {
        'name': 'Alphabet Inc.',
        'description': '구글 검색, 유튜브, AI (Gemini)',
        'sector': 'Communication Services',
        'industry': 'Internet Content & Information',
        'type': 'Stock'
    },
    'AMZN': {
        'name': 'Amazon.com Inc.',
        'description': '전자상거래, 클라우드(AWS) 1위',
        'sector': 'Consumer Cyclical',
        'industry': 'Internet Retail',
        'type': 'Stock'
    },
    'NVDA': {
        'name': 'NVIDIA Corporation',
        'description': 'AI 반도체(GPU) 독점적 지배자',
        'sector': 'Technology',
        'industry': 'Semiconductors',
        'type': 'Stock'
    },
    'META': {
        'name': 'Meta Platforms Inc.',
        'description': '페이스북, 인스타그램, AI(Llama)',
        'sector': 'Communication Services',
        'industry': 'Internet Content & Information',
        'type': 'Stock'
    },
    'TSLA': {
        'name': 'Tesla Inc.',
        'description': '전기차, 자율주행, 로봇',
        'sector': 'Consumer Cyclical',
        'industry': 'Auto Manufacturers',
        'type': 'Stock'
    },
    'COIN': {
        'name': 'Coinbase Global Inc.',
        'description': '미국 1위 암호화폐 거래소, 규제 준수',
        'sector': 'Financial Services',
        'industry': 'Cryptocurrency Exchange',
        'type': 'Stock'
    },
    'BTC-USD': {
        'name': 'Bitcoin',
        'description': '디지털 골드, 탈중앙화 화폐, 가치 저장 수단',
        'sector': 'Cryptocurrency',
        'industry': 'Digital Assets',
        'type': 'Crypto'
    }
}
# ==================== 핵심 함수 (순서 중요) ====================

@st.cache_data(ttl=3600)
def get_current_quarter_start():
    """
    현재 분기 시작일 계산 (날짜 보정 로직 포함)
    - 1월 초(15일 이전)인 경우 데이터 확보를 위해 작년 4분기로 설정
    """
    now = datetime.now()
    
    # 1. 일단 현재 분기 시작일 계산
    quarter = (now.month - 1) // 3
    quarter_start_month = quarter * 3 + 1
    current_q_start = datetime(now.year, quarter_start_month, 1)
    
    # 2. 현재 날짜와 분기 시작일 차이 계산
    days_diff = (now - current_q_start).days
    
    # 3. 분기 시작한 지 15일 미만이면 이전 분기로 설정 (데이터 부족 방지)
    if days_diff < 15:
        if quarter_start_month == 1: # 1월이면 작년 10월로
            return datetime(now.year - 1, 10, 1)
        else: # 아니면 3달 전으로
            return datetime(now.year, quarter_start_month - 3, 1)
            
    return current_q_start

def calculate_anchored_vwap(df):
    """
    Anchored VWAP 계산 함수 (이 함수가 누락되어 에러가 발생했음)
    """
    df = df.copy()
    # 고가, 저가, 종가의 평균 (Typical Price)
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    # 거래량 가중 가격
    df['TP_Volume'] = df['Typical_Price'] * df['Volume']
    # 누적 값 계산
    df['Cumulative_TP_Volume'] = df['TP_Volume'].cumsum()
    df['Cumulative_Volume'] = df['Volume'].cumsum()
    # VWAP 도출
    df['Anchored_VWAP'] = df['Cumulative_TP_Volume'] / df['Cumulative_Volume']
    return df

@st.cache_data(ttl=1800)
def get_quarterly_vwap_analysis(ticker):
    """분기별 Anchored VWAP 분석 (시가총액 수집 로직 개선)"""
    # 1. 날짜 계산
    try:
        quarter_start = get_current_quarter_start()
        end_date = datetime.now()
    except Exception as e:
        st.error(f"❌ [{ticker}] 날짜 계산 오류: {e}")
        return None

    # 2. 데이터 가져오기 (yfinance, rate limit 대응 retry)
    df = None
    for attempt in range(3):
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=quarter_start, end=end_date, auto_adjust=True)

            if df.empty:
                df = stock.history(start=quarter_start - timedelta(days=10), end=end_date, auto_adjust=True)
            break
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)  # 1초, 2초 대기 후 재시도
            else:
                st.warning(f"⚠️ [{ticker}] 데이터 수집 실패 (건너뜀): {e}")
                return None

    if df is None or df.empty:
        print(f"[{ticker}] 데이터 수집 실패")
        return None

    # 3. VWAP 및 지표 계산
    try:
        if len(df) < 1: return None
            
        df = calculate_anchored_vwap(df)
        
        current_price = df['Close'].iloc[-1]
        current_vwap = df['Anchored_VWAP'].iloc[-1]
        len_df = len(df)
        
        above_vwap_ratio = (df['Close'] > df['Anchored_VWAP']).sum() / len_df * 100
        
        # 추세 강도
        recent_20 = df['Close'].tail(min(20, len_df))
        if len(recent_20) > 1:
            uptrend_strength = (recent_20.diff() > 0).sum() / len(recent_20) * 100
        else:
            uptrend_strength = 50
            
        # 거래량 비율
        recent_vol = df['Volume'].tail(5).mean() if len_df >= 5 else df['Volume'].mean()
        avg_vol = df['Volume'].mean()
        volume_ratio = recent_vol / avg_vol if avg_vol > 0 else 1.0

        # --- [수정 핵심] 시가총액 및 정보 가져오기 로직 개선 ---
        company_name = ticker
        sector = 'N/A'
        market_cap = 0
        
        try:
            # 1단계: fast_info 사용 (가장 빠르고 정확함)
            if hasattr(stock, 'fast_info'):
                market_cap = stock.fast_info.get('market_cap', 0)
            
            # 2단계: fast_info 실패 시 info 시도 (구형 호환)
            if market_cap == 0:
                info = stock.info
                market_cap = info.get('marketCap', 0)
                company_name = info.get('longName', ticker)
                sector = info.get('sector', 'N/A')
            else:
                # fast_info로 시총은 구했으나 이름/섹터가 필요할 때
                # info는 느리므로 필요할 때만 호출하거나 MAG9_ASSETS 딕셔너리 활용 권장
                # 여기서는 기존 구조 유지를 위해 그대로 둠 (속도 저하 가능성 있음)
                pass

        except Exception as e:
            print(f"Info fetch error for {ticker}: {e}")

        # MAG9_ASSETS 딕셔너리에서 이름/섹터 정보 보완 (API 호출 최소화)
        if ticker in MAG9_ASSETS:
            company_name = MAG9_ASSETS[ticker]['name']
            sector = MAG9_ASSETS[ticker]['sector']

        quarter_start_price = df['Close'].iloc[0]
        quarter_return = ((current_price - quarter_start_price) / quarter_start_price * 100)
        q_num = (quarter_start.month - 1) // 3 + 1
        
        return {
            'Ticker': ticker,
            'Company': company_name,
            'Sector': sector,
            'Quarter': f'{quarter_start.year} Q{q_num}',
            'Trading_Days': len_df,
            'Current_Price': float(current_price),
            'Anchored_VWAP': float(current_vwap),
            'Quarter_Return_%': float(quarter_return),
            'Price_vs_VWAP_%': float((current_price - current_vwap) / current_vwap * 100),
            'Above_VWAP_Days_%': float(above_vwap_ratio),
            'Uptrend_Strength_%': float(uptrend_strength),
            'Volume_Ratio': float(volume_ratio),
            'Is_Above_VWAP': bool(current_price > current_vwap),
            'Market_Cap': float(market_cap) if market_cap else 0.0 # None 방지
        }

    except Exception as e:
        st.error(f"❌ [{ticker}] 지표 계산 중 오류: {e}")
        return None

def calculate_buy_score(row):
    """매수 신호 점수 계산"""
    score = 0
    if row['Is_Above_VWAP']:
        score += 30

    price_diff = row['Price_vs_VWAP_%']
    if 0 < price_diff <= 5:
        score += 20
    elif 5 < price_diff <= 10:
        score += 10
    elif price_diff > 10:
        score += 5

    if row['Above_VWAP_Days_%'] >= 80:
        score += 20
    elif row['Above_VWAP_Days_%'] >= 60:
        score += 15
    elif row['Above_VWAP_Days_%'] >= 40:
        score += 10

    if row['Uptrend_Strength_%'] >= 60:
        score += 15
    elif row['Uptrend_Strength_%'] >= 50:
        score += 10

    if row['Volume_Ratio'] >= 1.2:
        score += 15
    elif row['Volume_Ratio'] >= 1.0:
        score += 10

    return min(score, 100)

@st.cache_data(ttl=3600)
def get_comprehensive_fundamental(ticker):
    """상세 펀더멘털 분석"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info

        if ticker == 'BTC-USD':
            return None

        def safe_get(key, default=None):
            value = info.get(key)
            return value if value not in [None, 'N/A'] else default

        return {
            'Ticker': ticker,
            'PER': safe_get('trailingPE', 0),
            'PBR': safe_get('priceToBook', 0),
            'ROE_%': safe_get('returnOnEquity', 0) * 100 if safe_get('returnOnEquity') else 0,
            '영업이익률_%': safe_get('operatingMargins', 0) * 100 if safe_get('operatingMargins') else 0,
            '부채비율_%': safe_get('debtToEquity', 0),
            '매출성장률_%': safe_get('revenueGrowth', 0) * 100 if safe_get('revenueGrowth') else 0
        }

    except Exception as e:
        return None

@st.cache_data(ttl=86400)
def get_5year_growth_metrics(ticker):
    """5개년 성장률 분석"""
    try:
        stock = yf.Ticker(ticker)
        financials = stock.financials
        cashflow = stock.cashflow

        if financials.empty:
            return None

        years = financials.columns[:5] if len(financials.columns) >= 5 else financials.columns

        growth_data = {
            'Ticker': ticker,
            'Years': [year.year for year in years],
            'Revenue': [],
            'Operating_Income': [],
            'Net_Income': [],
            'Free_Cash_Flow': []
        }

        if 'Total Revenue' in financials.index:
            growth_data['Revenue'] = financials.loc['Total Revenue', years].tolist()

        if 'Operating Income' in financials.index:
            growth_data['Operating_Income'] = financials.loc['Operating Income', years].tolist()

        if 'Net Income' in financials.index:
            growth_data['Net_Income'] = financials.loc['Net Income', years].tolist()

        if 'Free Cash Flow' in cashflow.index:
            growth_data['Free_Cash_Flow'] = cashflow.loc['Free Cash Flow', years].tolist()

        return growth_data

    except Exception as e:
        return None

def calculate_cagr(start_value, end_value, years):
    """CAGR 계산"""
    if start_value <= 0 or end_value <= 0 or years <= 0:
        return 0
    try:
        cagr = (pow(end_value / start_value, 1 / years) - 1) * 100
        return round(cagr, 2)
    except:
        return 0

# ==================== AI 분석 함수 (Dual Engine 고도화) ====================

def get_market_summary_prompt(df_results):
    """시장 종합 분석용 프롬프트 생성"""
    
    # 데이터 집계
    total_count = len(df_results)
    above_vwap = len(df_results[df_results['Is_Above_VWAP'] == True])
    avg_score = df_results['Buy_Signal_Score'].mean()
    
    top_performers = df_results.head(3)['Ticker'].tolist()
    weak_performers = df_results.tail(3)['Ticker'].tolist()
    
    # 주식 vs 코인 성과 비교
    stock_return = df_results[df_results['Type'] == 'Stock']['Quarter_Return_%'].mean()
    crypto_return = df_results[df_results['Type'] == 'Crypto']['Quarter_Return_%'].mean() # BTC
    
    prompt = f"""
    당신은 월스트리트의 수석 전략가(Chief Strategist)입니다. 
    아래 'MAG 9 (미국 빅테크 7 + 코인베이스 + 비트코인)' 시장 데이터를 바탕으로 거시적인 시장 분석 리포트를 작성하세요.

    ## 시장 데이터 요약
    - 분석 시점: {datetime.now().strftime('%Y-%m-%d')}
    - 전체 자산 수: {total_count}개 중 {above_vwap}개가 VWAP(추세) 위에 있음 (강세장 여부 판단 기준)
    - 평균 매수 강도 점수: {avg_score:.1f}/100
    - 주식 평균 수익률: {stock_return:.2f}% vs 비트코인 수익률: {crypto_return:.2f}%
    - 선도주(Top 3): {', '.join(top_performers)}
    - 약세주(Bottom 3): {', '.join(weak_performers)}

    ## 분석 요구사항 (반드시 한국어로 작성)
    1. **시장 국면 진단**: 현재 시장이 위험 선호(Risk-On)인지 회피(Risk-Off)인지 진단하고 근거를 제시하세요.
    2. **자금 흐름 분석**: 빅테크(주식)와 암호화폐 간의 자금 이동이나 디커플링 현상이 보이는지 분석하세요.
    3. **섹터 로테이션**: 선도주와 약세주를 통해 현재 시장이 어떤 테마(예: AI, 금리 인하 기대 등)에 주목하는지 설명하세요.
    4. **포트폴리오 전략**: 현재 시점에서 주식과 암호화폐의 이상적인 비중 조절(Rebalancing) 의견을 제시하세요.
    
    결론은 명확하고 직설적으로 작성하세요.
    """
    return prompt

def get_single_stock_prompt(ticker_data, fundamental_data, cagr_data):
    """개별 종목 Deep Dive용 프롬프트 생성"""
    ticker = ticker_data['Ticker']
    
    prompt = f"""
    당신은 전설적인 퀀트 트레이더이자 펀더멘털 분석가입니다.
    다음 데이터를 바탕으로 '{ticker}'에 대한 심층 투자 보고서를 작성하세요.

    ## 1. 기술적 분석 (Anchored VWAP 기준)
    - 현재가: ${ticker_data['Current_Price']}
    - Anchored VWAP: ${ticker_data['Anchored_VWAP']} (지지/저항 라인)
    - VWAP 괴리율: {ticker_data['Price_vs_VWAP_%']:+.2f}%
    - 추세 강도: {ticker_data['Uptrend_Strength_%']}/100
    - 매수 신호 점수: {ticker_data['Buy_Signal_Score']}/100

    ## 2. 펀더멘털 및 성장성
    {fundamental_data if fundamental_data else "정보 없음"}
    {cagr_data if cagr_data else "성장률 데이터 없음"}

    ## 분석 요구사항 (반드시 한국어로 작성)
    1. **트레이딩 셋업 (시나리오)**:
       - **Bull Case (상승)**: 진입 적정가와 1차 목표가를 구체적 숫자로 제시.
       - **Bear Case (하락)**: VWAP 이탈 시 손절(Stop Loss) 라인과 하방 지지선 제시.
    
    2. **펀더멘털/성장성 평가**:
       - 현재 밸류에이션(PER, PBR 등)이 성장률(CAGR) 대비 정당화 가능한지 분석 (PEG 관점).
    
    3. **리스크 요인**:
       - 이 종목이 가진 가장 치명적인 약점 2가지를 지적하세요.
    
    4. **최종 투자의견**: 
       - [강력 매수 / 분할 매수 / 관망 / 매도] 중 하나를 선택하고 이유를 한 문장으로 요약하세요.
    """
    return prompt

def call_openai_api(prompt):
    if not OPENAI_ENABLED: return "OpenAI API 키가 없습니다."
    try:
        response = OPENAI_CLIENT.chat.completions.create(
            model=OPENAI_MODEL_MARKET,
            messages=[
                {"role": "system", "content": "너는 냉철하고 분석적인 전문 투자 자문가다. 서론/본론/결론 형식을 갖춰라."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"OpenAI 오류: {str(e)}"

def call_gemini_api(prompt):
    if not GEMINI_ENABLED: return "Gemini API 키가 없습니다."
    try:
        model = genai.GenerativeModel('gemini-2.5-flash') # 모델명 확인 필요 (gemini-pro 또는 gemini-1.5-flash 권장)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Gemini 오류: {str(e)}"


# ==================== 메인 앱 ====================
st.title("🚀 MAG 9 (MAG 7 + COIN + BTC) AI 종합 분석")
st.markdown("### 📊 Anchored VWAP 기반 투자 전략 시스템")

quarter_start = get_current_quarter_start()
quarter_num = (quarter_start.month - 1) // 3 + 1

st.info(f"""
📍 **분석 기준**: Anchored VWAP ({quarter_start.year} Q{quarter_num})  
📅 **Anchor Point**: {quarter_start.strftime('%Y-%m-%d')}  
🌟 **분석 대상**: MAG 7 + COINBASE + BITCOIN (9개 자산)
""")

# ==================== 기술적 분석 ====================
# ==================== 기술적 분석 ====================
if show_technical:
    st.markdown("---")
    st.header("📈 기술적 분석 (Anchored VWAP)")
    
    with st.spinner("MAG 9 데이터 수집 중..."):
        results = []
        all_tickers = list(MAG9_ASSETS.keys())
        
        progress_bar = st.progress(0)
        for idx, ticker in enumerate(all_tickers):
            # 1. VWAP 분석 데이터 가져오기 (종목 간 딜레이로 rate limit 방지)
            if idx > 0:
                time.sleep(0.5)
            result = get_quarterly_vwap_analysis(ticker)
            
            # 2. 데이터가 성공적으로 수집되었을 경우, 자산 정보(Type, Description 등) 병합
            if result:
                asset_info = MAG9_ASSETS.get(ticker, {})
                result['Type'] = asset_info.get('type', 'Stock')
                result['Description'] = asset_info.get('description', '')
                # Market_Cap이 yfinance info에 없는 경우를 대비 (Crypto 등)
                if 'Market_Cap' not in result:
                     result['Market_Cap'] = 0 
                results.append(result)
            
            progress_bar.progress((idx + 1) / len(all_tickers))
        
        progress_bar.empty()
        
        # [수정 핵심 1] 결과 리스트가 비어있는지 확인 (방어 코드)
        if not results:
            st.error("❌ 데이터를 가져오지 못했습니다.")
            st.warning("""
            **가능한 원인:**
            1. Yahoo Finance API 호출 제한 (잠시 후 다시 시도하세요)
            2. 인터넷 연결 문제
            3. 연초(1월 초)라 거래일 데이터 부족 (get_quarterly_vwap_analysis 함수 확인 필요)
            """)
            st.stop() # 이후 코드 실행 중단

        df_results = pd.DataFrame(results)

        # [수정 핵심 2] 데이터프레임 생성 후 점수 계산
        try:
            df_results['Buy_Signal_Score'] = df_results.apply(calculate_buy_score, axis=1)
            df_results = df_results.sort_values('Buy_Signal_Score', ascending=False)
            
            # 시가총액 조(Trillion) 단위 변환 (에러 방지용 fillna 추가)
            df_results['Market_Cap'] = df_results['Market_Cap'].fillna(0)
            df_results['Market_Cap_Trillion'] = (df_results['Market_Cap'] / 1e12).round(3)
            
        except Exception as e:
            st.error(f"데이터 계산 중 오류 발생: {e}")
            st.stop()

    st.success("✓ 데이터 수집 완료!")
    
    # ... (이하 코드 동일)


    
    # 상위 3개 종목 카드
    st.subheader("🏆 TOP 3 추천 종목")
    
    cols = st.columns(3)
    for idx in range(min(3, len(df_results))):
        row = df_results.iloc[idx]
        
        with cols[idx]:
            if row['Type'] == 'Crypto':
                icon = "₿"
                color = "orange"
            elif row['Ticker'] == 'COIN':
                icon = "💰"
                color = "purple"
            else:
                icon = "📈"
                color = "blue"
            
            medal = ["🥇", "🥈", "🥉"][idx]
            
            st.markdown(f"""
            <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid {color}">
                <h3>{medal} {icon} {row['Ticker']}</h3>
                <p><strong>{row['Company']}</strong></p>
                <p style="font-size: 0.9em; color: #666;">{row['Description']}</p>
                <hr>
                <p><strong>현재가:</strong> ${row['Current_Price']:,.2f}</p>
                <p><strong>시가총액:</strong> ${row['Market_Cap_Trillion']:.2f}T</p>
                <p><strong>분기 수익률:</strong> <span style="color: {'green' if row['Quarter_Return_%'] > 0 else 'red'};">{row['Quarter_Return_%']:+.2f}%</span></p>
                <p><strong>매수 점수:</strong> {row['Buy_Signal_Score']}/100</p>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # 전체 데이터 테이블
    st.subheader("📊 전체 종목 분석 결과")
    
    display_cols = ['Ticker', 'Company', 'Type', 'Current_Price', 'Anchored_VWAP',
                    'Price_vs_VWAP_%', 'Quarter_Return_%', 'Market_Cap_Trillion', 'Buy_Signal_Score']
    
    st.dataframe(
        df_results[display_cols].style.background_gradient(
            subset=['Buy_Signal_Score'],
            cmap='RdYlGn'
        ).background_gradient(
            subset=['Quarter_Return_%'],
            cmap='RdYlGn'
        ).format({
            'Current_Price': '${:.2f}',
            'Anchored_VWAP': '${:.2f}',
            'Price_vs_VWAP_%': '{:+.2f}%',
            'Quarter_Return_%': '{:+.2f}%',
            'Market_Cap_Trillion': '${:.2f}T'
        }),
        use_container_width=True
    )
    
    # 차트
    st.subheader("📈 시각화 분석")
    
    tab1, tab2, tab3 = st.tabs(["매수 신호 점수", "VWAP 대비 가격", "시가총액 vs 수익률"])
    
    with tab1:
        colors_score = []
        for _, row in df_results.iterrows():
            if row['Buy_Signal_Score'] >= 80:
                color = '#00C853'
            elif row['Buy_Signal_Score'] >= 60:
                color = '#FFD600'
            else:
                color = '#2196F3'
            
            if row['Type'] == 'Crypto':
                color = '#FF9800'
            elif row['Ticker'] == 'COIN':
                color = '#9C27B0'
            
            colors_score.append(color)
        
        fig1 = go.Figure()
        fig1.add_trace(go.Bar(
            y=df_results['Ticker'],
            x=df_results['Buy_Signal_Score'],
            orientation='h',
            marker=dict(color=colors_score),
            text=df_results['Buy_Signal_Score'],
            textposition='auto'
        ))
        
        fig1.update_layout(
            title=f'매수 신호 점수 ({quarter_start.year} Q{quarter_num})',
            xaxis_title='매수 신호 점수',
            yaxis_title='종목',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig1, use_container_width=True)
    
    with tab2:
        colors_vwap = []
        for _, row in df_results.iterrows():
            if row['Price_vs_VWAP_%'] > 0:
                color = 'green' if row['Type'] != 'Crypto' and row['Ticker'] != 'COIN' else ('#FF9800' if row['Type'] == 'Crypto' else '#9C27B0')
            else:
                color = 'red'
            colors_vwap.append(color)
        
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            y=df_results['Ticker'],
            x=df_results['Price_vs_VWAP_%'],
            orientation='h',
            marker=dict(color=colors_vwap),
            text=df_results['Price_vs_VWAP_%'].round(1),
            textposition='auto'
        ))
        
        fig2.add_vline(x=0, line_dash="dash", line_color="black", line_width=2)
        
        fig2.update_layout(
            title=f'VWAP 대비 가격 위치 ({quarter_start.year} Q{quarter_num})',
            xaxis_title='VWAP 대비 차이 (%)',
            yaxis_title='종목',
            height=500,
            template='plotly_white'
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab3:
        fig3 = px.scatter(
            df_results,
            x='Market_Cap_Trillion',
            y='Quarter_Return_%',
            size='Buy_Signal_Score',
            color='Type',
            hover_data=['Ticker', 'Company'],
            text='Ticker',
            color_discrete_map={'Stock': '#2196F3', 'Crypto': '#FF9800'},
            title=f'시가총액 vs 분기 수익률 ({quarter_start.year} Q{quarter_num})'
        )
        
        fig3.update_traces(textposition='top center')
        fig3.update_layout(height=500, template='plotly_white')
        
        st.plotly_chart(fig3, use_container_width=True)

# ==================== 펀더멘털 분석 ====================
if show_fundamental:
    st.markdown("---")
    st.header("💼 펀더멘털 분석 (6개 지표)")
    
    with st.spinner("펀더멘털 데이터 수집 중..."):
        fundamental_data = []
        stock_tickers = [t for t in all_tickers if MAG9_ASSETS[t]['type'] == 'Stock']
        
        for ticker in stock_tickers:
            fund_data = get_comprehensive_fundamental(ticker)
            if fund_data:
                fundamental_data.append(fund_data)
        
        df_fundamental = pd.DataFrame(fundamental_data)
    
    if not df_fundamental.empty:
        st.success("✓ 펀더멘털 데이터 수집 완료!")
        
        st.dataframe(
            df_fundamental.style.format({
                'PER': '{:.2f}',
                'PBR': '{:.2f}',
                'ROE_%': '{:.2f}%',
                '영업이익률_%': '{:.2f}%',
                '부채비율_%': '{:.2f}%',
                '매출성장률_%': '{:.2f}%'
            }),
            use_container_width=True
        )
        
        # 펀더멘털 차트
        fig_fund = make_subplots(
            rows=2, cols=3,
            subplot_titles=('PER (낮을수록 저평가)', 'PBR (낮을수록 저평가)', 'ROE (%)',
                           '영업이익률 (%)', '부채비율 (낮을수록 양호)', '매출성장률 (%)'),
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        metrics = [
            ('PER', 1, 1, '#3498db'),
            ('PBR', 1, 2, '#2ecc71'),
            ('ROE_%', 1, 3, '#f39c12'),
            ('영업이익률_%', 2, 1, '#e74c3c'),
            ('부채비율_%', 2, 2, '#9b59b6'),
            ('매출성장률_%', 2, 3, '#1abc9c')
        ]
        
        for metric, row, col, color in metrics:
            fig_fund.add_trace(
                go.Bar(
                    x=df_fundamental['Ticker'],
                    y=df_fundamental[metric],
                    name=metric,
                    marker_color=color,
                    text=df_fundamental[metric].round(2),
                    textposition='auto',
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig_fund.update_layout(
            title_text=f'펀더멘털 6개 지표 비교 ({quarter_start.year} Q{quarter_num})',
            height=800,
            showlegend=False,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_fund, use_container_width=True)

# ==================== 5개년 성장률 분석 ====================
if show_growth:
    st.markdown("---")
    st.header("📈 5개년 성장률 분석 (CAGR)")
    
    with st.spinner("5개년 재무 데이터 수집 중..."):
        all_growth_data = []
        stock_tickers = [t for t in all_tickers if MAG9_ASSETS[t]['type'] == 'Stock']
        
        for ticker in stock_tickers:
            growth_data = get_5year_growth_metrics(ticker)
            if growth_data:
                all_growth_data.append(growth_data)
    
    if all_growth_data:
        st.success(f"✓ 총 {len(all_growth_data)}개 종목 데이터 수집 완료!")
        
        # CAGR 계산
        cagr_summary = []
        
        for data in all_growth_data:
            ticker = data['Ticker']
            years_count = len(data['Years']) - 1
            
            if years_count <= 0:
                continue
            
            cagr_data = {'Ticker': ticker}
            
            metrics_cagr = [
                ('Revenue', '매출_CAGR_%'),
                ('Net_Income', '순이익_CAGR_%'),
                ('Operating_Income', '영업이익_CAGR_%'),
                ('Free_Cash_Flow', 'FCF_CAGR_%')
            ]
            
            for metric_key, cagr_key in metrics_cagr:
                if data[metric_key] and len(data[metric_key]) >= 2:
                    start_val = data[metric_key][-1]
                    end_val = data[metric_key][0]
                    if start_val > 0 and end_val > 0:
                        cagr_data[cagr_key] = calculate_cagr(start_val, end_val, years_count)
                    else:
                        cagr_data[cagr_key] = None
                else:
                    cagr_data[cagr_key] = None
            
            cagr_data['분석기간'] = f"{data['Years'][-1]}-{data['Years'][0]}"
            cagr_summary.append(cagr_data)
        
        df_cagr = pd.DataFrame(cagr_summary)
        
        st.dataframe(
            df_cagr.style.format({
                '매출_CAGR_%': '{:.2f}%',
                '순이익_CAGR_%': '{:.2f}%',
                '영업이익_CAGR_%': '{:.2f}%',
                'FCF_CAGR_%': '{:.2f}%'
            }, na_rep='N/A'),
            use_container_width=True
        )
        
        # CAGR 차트
        fig_cagr = go.Figure()
        
        cagr_metrics = ['매출_CAGR_%', '순이익_CAGR_%', '영업이익_CAGR_%', 'FCF_CAGR_%']
        colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
        
        for idx, metric in enumerate(cagr_metrics):
            values = []
            tickers = []
            
            for _, row in df_cagr.iterrows():
                val = row[metric]
                if pd.notna(val):
                    values.append(val)
                    tickers.append(row['Ticker'])
            
            if values:
                fig_cagr.add_trace(go.Bar(
                    name=metric.replace('_CAGR_%', ''),
                    x=tickers,
                    y=values,
                    text=[f"{v:.1f}%" for v in values],
                    textposition='auto',
                    marker_color=colors[idx]
                ))
        
        fig_cagr.update_layout(
            title='5개년 CAGR 비교 (4개 지표)',
            xaxis_title='종목',
            yaxis_title='CAGR (%)',
            barmode='group',
            height=600,
            template='plotly_white'
        )
        
        st.plotly_chart(fig_cagr, use_container_width=True)


# ==================== AI Deep Dive 분석 (Dual Engine) ====================
if show_ai:
    st.markdown("---")
    st.header("🤖 AI Deep Dive (Dual Engine)")
    st.markdown("OpenAI(GPT-4)와 Google Gemini가 동시에 분석하여 교차 검증합니다.")

    # 탭으로 분석 모드 분리
    tab_market, tab_stock = st.tabs(["🌍 종합 시장 분석 (Macro)", "🔍 개별 종목 심층 분석 (Micro)"])

    # 1. 종합 시장 분석 탭
    with tab_market:
        st.subheader("📊 MAG 9 시장 전체 브리핑")
        st.info("MAG 9 전체 데이터(주식+코인)를 기반으로 시장의 흐름과 자금 이동을 분석합니다.")
        
        if st.button("🚀 종합 시장 분석 실행", type="primary", use_container_width=True):
            market_prompt = get_market_summary_prompt(df_results)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 🧠 OpenAI (Strategist)")
                with st.spinner("OpenAI 분석 중..."):
                    openai_res = call_openai_api(market_prompt)
                    st.markdown(openai_res)
            
            with col2:
                st.markdown("### 🌟 Gemini (Analyst)")
                with st.spinner("Gemini 분석 중..."):
                    gemini_res = call_gemini_api(market_prompt)
                    st.markdown(gemini_res)

    # 2. 개별 종목 심층 분석 탭
    with tab_stock:
        st.subheader("🔍 개별 자산 Deep Dive")
        
        # 종목 선택 (점수 높은 순으로 정렬)
        sorted_tickers = df_results['Ticker'].tolist()
        selected_ticker = st.selectbox("분석할 종목을 선택하세요:", sorted_tickers)
        
        if st.button(f"🚀 {selected_ticker} 심층 분석 실행", type="primary", use_container_width=True):
            # 선택된 종목 데이터 준비
            row = df_results[df_results['Ticker'] == selected_ticker].iloc[0]
            
            # 펀더멘털 데이터
            fund_data = None
            if not df_fundamental.empty and selected_ticker in df_fundamental['Ticker'].values:
                fund_data = df_fundamental[df_fundamental['Ticker'] == selected_ticker].iloc[0].to_dict()
            
            # 성장률(CAGR) 데이터
            cagr_data = None
            if not df_cagr.empty and selected_ticker in df_cagr['Ticker'].values:
                cagr_data = df_cagr[df_cagr['Ticker'] == selected_ticker].iloc[0].to_dict()
            
            # 프롬프트 생성
            stock_prompt = get_single_stock_prompt(row.to_dict(), fund_data, cagr_data)
            
            # Dual Engine 실행
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### 🧠 OpenAI 분석 ({selected_ticker})")
                with st.spinner("OpenAI가 차트와 장부를 분석 중..."):
                    stock_res_openai = call_openai_api(stock_prompt)
                    st.success("OpenAI 분석 완료")
                    st.markdown(stock_res_openai)
            
            with col2:
                st.markdown(f"### 🌟 Gemini 분석 ({selected_ticker})")
                with st.spinner("Gemini가 데이터를 크로스체크 중..."):
                    stock_res_gemini = call_gemini_api(stock_prompt)
                    st.success("Gemini 분석 완료")
                    st.markdown(stock_res_gemini)


# ==================== 투자 전략 요약 ====================
st.markdown("---")
st.header("💡 투자 전략 종합 요약")

col1, col2, col3, col4 = st.columns(4)

with col1:
    above_vwap = df_results[df_results['Is_Above_VWAP'] == True]
    st.metric("VWAP 위 자산", f"{len(above_vwap)}개", f"{len(above_vwap)/len(df_results)*100:.0f}%")

with col2:
    strong_buy = df_results[df_results['Buy_Signal_Score'] >= 80]
    st.metric("강력 매수 (80점↑)", f"{len(strong_buy)}개", 
              ", ".join(strong_buy['Ticker'].tolist()) if not strong_buy.empty else "없음")

with col3:
    good_buy = df_results[(df_results['Buy_Signal_Score'] >= 60) & (df_results['Buy_Signal_Score'] < 80)]
    st.metric("눌림목 대기 (60-80점)", f"{len(good_buy)}개",
              ", ".join(good_buy['Ticker'].tolist()) if not good_buy.empty else "없음")

with col4:
    watch = df_results[df_results['Buy_Signal_Score'] < 60]
    st.metric("관찰 필요 (60점 미만)", f"{len(watch)}개",
              ", ".join(watch['Ticker'].tolist()) if not watch.empty else "없음")

# 자산군별 평균 성과
st.markdown("### 📊 자산군별 평균 성과")

stocks_df = df_results[df_results['Type'] == 'Stock']
crypto_df = df_results[df_results['Type'] == 'Crypto']

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📈 주식 (MAG 7 + COIN)")
    st.write(f"평균 분기 수익률: **{stocks_df['Quarter_Return_%'].mean():.2f}%**")
    st.write(f"평균 매수 점수: **{stocks_df['Buy_Signal_Score'].mean():.1f}/100**")

with col2:
    if not crypto_df.empty:
        st.markdown("#### ₿ 암호화폐 (BTC)")
        st.write(f"분기 수익률: **{crypto_df['Quarter_Return_%'].iloc[0]:.2f}%**")
        st.write(f"매수 점수: **{crypto_df['Buy_Signal_Score'].iloc[0]}/100**")

# 투자 가이드
st.markdown("### 📋 AI 기반 투자 가이드")

st.info("""
**1. 💚 강력 매수 (80점 이상)**
- VWAP 위에서 안정적, AI 분석 긍정적
- 즉시 매수 검토 (단, VWAP +5% 이상이면 눌림목 대기)

**2. 💛 눌림목 대기 (60-80점)**
- 펀더멘털 우수, VWAP 근처 조정 시 매수
- 손절라인: VWAP -2%

**3. 💙 관찰 필요 (60점 미만)**
- VWAP 돌파 확인 후 재검토
- AI 분석 리스크 요소 확인 필수

**4. 🤖 AI Deep Dive 활용**
- OpenAI와 Gemini의 교차 분석 활용
- 양측 AI가 동의하는 전략에 가중치 부여

**5. 💰 COIN & ₿ BTC 특별 전략**
- COIN: BTC 가격 상관관계, 규제 리스크 모니터링
- BTC: 변동성 높음, 포트폴리오 5-10% 권장, 장기 관점
""")

# 푸터
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p>📊 MAG 9 AI 종합 분석 시스템 | Powered by Streamlit</p>
    <p>⚠️ 이 분석은 투자 조언이 아닙니다. 투자 결정은 본인의 책임입니다.</p>
</div>
""", unsafe_allow_html=True)
