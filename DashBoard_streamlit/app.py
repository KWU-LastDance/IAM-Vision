import pandas as pd
import numpy as np
import statsmodels.api as sm
from pmdarima import auto_arima
import streamlit as st
import altair as alt
from datetime import datetime, timedelta

# 최적의 ARIMA 파라미터를 찾는 함수
def find_best_arima_params(data):
    model = auto_arima(data, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
    return model.order

# 예측 데이터를 생성하는 함수
def generate_forecast(data, best_params):
    last_date = data['date'].max()
    next_date = last_date + pd.Timedelta(days=1)
    forecast_data = {'date': [next_date]}

    for quality in ['special', 'good', 'bad']:
        best_order = best_params[quality]
        model = sm.tsa.ARIMA(data[quality], order=best_order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        forecast = np.round(forecast).astype(int)  # 정수로 변환
        forecast_data[quality] = forecast

    forecast_df = pd.DataFrame(forecast_data)
    return forecast_df

# 변화를 계산하는 함수
def calculate_percentage_change(current_value, previous_value):
    if previous_value == 0:
        return "처음입니다."
    percentage_change = ((current_value - previous_value) / previous_value) * 100
    return f"{percentage_change:.2f}%"

def main():
    file_path = './apple_stock.csv'
    data = pd.read_csv(file_path)

    # 데이터 전처리
    data['date'] = pd.to_datetime(data['date'])

    # 최적의 ARIMA 파라미터 찾기
    best_params = {}
    for quality in ['special', 'good', 'bad']:
        best_order = find_best_arima_params(data[quality])
        best_params[quality] = best_order

    # 예측 데이터 생성
    forecast_data = generate_forecast(data, best_params)

    # Streamlit 앱 설정
    st.set_page_config(page_title="Dashboard", layout="wide", initial_sidebar_state="auto")

    # 날짜 선택
    st.sidebar.header("날짜 범위 선택")
    start_date = st.sidebar.date_input("시작 날짜", value=data['date'].min())
    end_date = st.sidebar.date_input("종료 날짜", value=data['date'].max())

    # 선택한 날짜 범위에 따라 데이터 필터링
    filtered_data = data[(data['date'] >= pd.to_datetime(start_date)) & (data['date'] <= pd.to_datetime(end_date))]

    # 변화 계산을 위해 이전 날짜 범위 계산
    previous_start_date = start_date - (end_date - start_date + timedelta(days=1))
    previous_end_date = start_date - timedelta(days=1)
    previous_filtered_data = data[(data['date'] >= pd.to_datetime(previous_start_date)) & (data['date'] <= pd.to_datetime(previous_end_date))]

    # 현재 갯수 계산
    current_special_sum = filtered_data['special'].sum()
    current_good_sum = filtered_data['good'].sum()
    current_bad_sum = filtered_data['bad'].sum()

    # 이전 기간의 갯수 계산
    previous_special_sum = previous_filtered_data['special'].sum()
    previous_good_sum = previous_filtered_data['good'].sum()
    previous_bad_sum = previous_filtered_data['bad'].sum()


    st.title("Dashboard")
    st.caption(f"{end_date.strftime('%Y년 %m월 %d일')} 기준")
    col1, col2, col3 = st.columns(3)

    # 현재 갯수와 이전 기간 대비 변화율을 나타냄
    col1.metric("Special", f'{current_special_sum}', f'{calculate_percentage_change(current_special_sum, previous_special_sum)}')
    col2.metric("Good", f'{current_good_sum}', f'{calculate_percentage_change(current_good_sum, previous_good_sum)}')
    col3.metric("Bad", f'{current_bad_sum}', f'{calculate_percentage_change(current_bad_sum, previous_bad_sum)}')

    st.divider()
    
    col1, col2 = st.columns([0.6, 0.4])

    with col1:
        # 선택한 날짜 범위에 따른 품질별 사과 갯수 변화 시각화
        st.subheader("Apple Quantity Distribution by Quality")
        line_chart = alt.Chart(filtered_data.melt(id_vars='date', value_vars=['special', 'good', 'bad'])).mark_line().encode(
            x=alt.X('date:T', title='날짜'),
            y=alt.Y('value:Q', title='수량'),
            color='variable:N',
            tooltip=['date:T', 'value:Q', 'variable:N']
        ).properties(
            height=300
        ).interactive()
        col1.altair_chart(line_chart, use_container_width=True)

    # 품질별 재고 가장 많은날 구하기
    peak_days = []
    for quality in ['special', 'good', 'bad']:
        peak_day = filtered_data.loc[filtered_data[quality].idxmax(), ['date', quality]]
        peak_days.append({'quality': quality, 'date': peak_day['date'], 'quantity': peak_day[quality]})

    peak_data = pd.DataFrame(peak_days)

    # 품질별 재고 가장 많은 날을 막대 차트로 시각화
    with col2:
        st.subheader("Top Day for Apple Inventory")
        peak_chart = alt.Chart(peak_data).mark_bar().encode(
            x=alt.X('quantity:Q', title='수량'),
            y=alt.Y('date:T', title='날짜', sort='-x'),
            color='quality:N',
            tooltip=['date:T', 'quantity:Q', 'quality:N']
        ).properties(
            height=300
        ).interactive()
        col2.altair_chart(peak_chart, use_container_width=True)

    col1, col2 = st.columns(2)

    # 품질별 사과 수량 비율을 파이 차트로 시각화
    col1.subheader("Apple Quantity Ratio by Quality")
    pie_data = filtered_data[['special', 'good', 'bad']].sum().reset_index()
    pie_data.columns = ['quality', 'quantity']
    pie_chart = alt.Chart(pie_data).mark_arc().encode(
        theta=alt.Theta(field="quantity", type="quantitative"),
        color=alt.Color(field="quality", type="nominal")
    )
    col1.altair_chart(pie_chart, use_container_width=True)

    # 다음날 품질별 사과 수량 예측을 막대 차트로 시각화
    col2.subheader("Forecast of Tomorrow Apple Count")
    forecast_chart = alt.Chart(forecast_data.melt(id_vars='date', value_vars=['special', 'good', 'bad'])).mark_bar().encode(
        x=alt.X('value:Q', title='수량'),
        y=alt.Y('variable:N', title='품질'),
        color='variable:N'
    )
    col2.altair_chart(forecast_chart, use_container_width=True)

if __name__ == "__main__":
    main()
