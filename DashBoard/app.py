import pandas as pd
import numpy as np
import statsmodels.api as sm
from pmdarima import auto_arima
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

# 데이터 로드
file_path = 'apple_stock.csv'
data = pd.read_csv(file_path)

# 날짜 형식을 datetime으로 변환
data['date'] = pd.to_datetime(data['date'])

# 최적의 하이퍼파라미터 찾기 함수 정의
def find_best_arima_params(data):
    model = auto_arima(data, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
    return model.order

# 각 품질별로 최적의 하이퍼파라미터 찾기
best_params = {}
for quality in ['special', 'good', 'bad']:
    best_order = find_best_arima_params(data[quality])
    best_params[quality] = best_order
    print(f'Best order for {quality}: {best_order}')

# 예측 생성 함수 정의 (다음날 예측)
def generate_forecast(data, best_params):
    last_date = data['date'].max()
    next_date = last_date + pd.Timedelta(days=1)
    forecast_data = {'date': [next_date]}

    for quality in ['special', 'good', 'bad']:
        best_order = best_params[quality]
        model = sm.tsa.ARIMA(data[quality], order=best_order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        forecast = np.round(forecast).astype(int)  # 정수형으로 변환
        forecast_data[quality] = forecast

    forecast_df = pd.DataFrame(forecast_data)
    return forecast_df

# Dash 앱 초기화
app = Dash(__name__)

# 레이아웃 정의
app.layout = html.Div([
    dcc.DatePickerRange(
        id='date-picker-range',
        start_date=data['date'].min(),
        end_date=data['date'].max(),
        display_format='YYYY-MM-DD',
        style={'backgroundColor': '#E0F7FA', 'color': '#000000'}
    ),
    dcc.Graph(id='line-chart', style={'backgroundColor': '#E0F7FA'}),
    dcc.Graph(id='pie-chart', style={'backgroundColor': '#E0F7FA'}),
    dcc.Graph(id='forecast-chart', style={'backgroundColor': '#E0F7FA'})
], style={'backgroundColor': '#E0F7FA'})

# 콜백 함수 정의
@app.callback(
    [Output('line-chart', 'figure'),
     Output('pie-chart', 'figure'),
     Output('forecast-chart', 'figure')],
    [Input('date-picker-range', 'start_date'),
     Input('date-picker-range', 'end_date')]
)
def update_charts(start_date, end_date):
    if not start_date or not end_date:
        return {}, {}, {}

    # 선택한 날짜 범위에 따른 데이터 필터링
    filtered_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]

    # 라인 차트 생성
    line_fig = px.line(filtered_data, x='date', y=['special', 'good', 'bad'], 
                       labels={'value': '수량', 'variable': '품질'},
                       title='날짜별 품질별 사과 수량')
    line_fig.update_layout(xaxis_title='날짜', yaxis_title='수량', template='plotly', plot_bgcolor='#E0F7FA')

    # 원형 차트 생성
    pie_data = filtered_data[['special', 'good', 'bad']].sum().reset_index()
    pie_data.columns = ['quality', 'quantity']
    pie_fig = px.pie(pie_data, values='quantity', names='quality', title='품질별 사과 수량 비율')
    pie_fig.update_layout(template='plotly', plot_bgcolor='#E0F7FA')

    # 예측 데이터 생성
    forecast_data = generate_forecast(data, best_params)
    
    # 예측 데이터 막대그래프 생성
    forecast_fig = px.bar(forecast_data.melt(id_vars='date', value_vars=['special', 'good', 'bad']),
                          x='variable', y='value', color='variable', 
                          labels={'value': '수량', 'variable': '품질'},
                          title='다음날 품질별 사과 수량 예측')
    forecast_fig.update_layout(xaxis_title='품질', yaxis_title='수량', template='plotly', plot_bgcolor='#E0F7FA')

    return line_fig, pie_fig, forecast_fig

# 서버 실행
if __name__ == '__main__':
    app.run_server(debug=True)
