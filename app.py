import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
try:
    from prophet import Prophet
    PROPHET_OK = True
except Exception:
    PROPHET_OK = False

st.set_page_config(page_title="Hyper-Local Demand Forecasting", layout="wide")
st.title("Hyper-Local Demand Forecasting Engine")
st.caption("Synthetic quick-commerce dark store demand analysis and forecasting dashboard")

@st.cache_data
def load_data(path):
    return pd.read_csv(path, parse_dates=['timestamp'])

def make_features(df):
    data = df.copy().sort_values(['store_id','sku_id','timestamp'])
    data['hour'] = data['timestamp'].dt.hour
    data['dayofweek'] = data['timestamp'].dt.dayofweek
    data['day'] = data['timestamp'].dt.day
    data['weekofyear'] = data['timestamp'].dt.isocalendar().week.astype(int)
    data['month'] = data['timestamp'].dt.month
    data['is_weekend'] = (data['dayofweek'] >= 5).astype(int)
    for lag in [1, 2, 24]:
        data[f'lag_{lag}'] = data.groupby(['store_id','sku_id'])['quantity_sold'].shift(lag)
    for win in [3, 6, 12, 24]:
        data[f'roll_mean_{win}'] = data.groupby(['store_id','sku_id'])['quantity_sold'].transform(lambda s: s.shift(1).rolling(win, min_periods=1).mean())
    data = pd.get_dummies(data, columns=['store_id','sku_id','category','weather_condition'], drop_first=False)
    return data

def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    denom = np.where(np.array(y_true)==0, 1, np.array(y_true))
    mape = np.mean(np.abs((np.array(y_true)-np.array(y_pred))/denom))*100
    r2 = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}

df = load_data('synthetic_qcommerce_demand.csv')
st.sidebar.header('Filters')
store = st.sidebar.multiselect('Store', sorted(df['store_id'].unique()), default=sorted(df['store_id'].unique()))
sku = st.sidebar.multiselect('SKU', sorted(df['sku_id'].unique()), default=sorted(df['sku_id'].unique()))
flt = df[df['store_id'].isin(store) & df['sku_id'].isin(sku)].copy()

c1,c2,c3,c4 = st.columns(4)
c1.metric('Rows', len(flt))
c2.metric('Stores', flt['store_id'].nunique())
c3.metric('SKUs', flt['sku_id'].nunique())
c4.metric('Stockout rate %', round(flt['stockout_flag'].mean()*100,2))

st.subheader('Descriptive analysis')
hourly = flt.groupby(flt['timestamp'].dt.hour)['quantity_sold'].mean().reset_index(name='avg_qty')
st.plotly_chart(px.line(hourly, x='timestamp', y='avg_qty', labels={'timestamp':'Hour of day','avg_qty':'Average quantity sold'}), use_container_width=True)
daily = flt.groupby(flt['timestamp'].dt.date)['quantity_sold'].sum().reset_index(name='daily_qty')
st.plotly_chart(px.line(daily, x='timestamp', y='daily_qty', labels={'timestamp':'Date','daily_qty':'Daily quantity sold'}), use_container_width=True)
weather = flt.groupby('weather_condition')['quantity_sold'].agg(['mean','sum','count']).reset_index()
st.plotly_chart(px.bar(weather, x='weather_condition', y='mean', title='Average sales by weather'), use_container_width=True)

st.subheader('Diagnostic analysis')
pivot = flt.pivot_table(index=flt['timestamp'].dt.hour, columns='weather_condition', values='quantity_sold', aggfunc='mean').reset_index()
st.plotly_chart(px.line(pivot, x='timestamp', y=[c for c in pivot.columns if c!='timestamp'], labels={'timestamp':'Hour'}), use_container_width=True)
holiday_cmp = flt.groupby('public_holiday_flag')['quantity_sold'].mean().reset_index()
st.plotly_chart(px.bar(holiday_cmp, x='public_holiday_flag', y='quantity_sold', title='Holiday vs non-holiday demand'), use_container_width=True)

st.subheader('Modeling')
feat = make_features(df[['timestamp','store_id','sku_id','quantity_sold','price_at_sale','stockout_flag','weather_condition','temperature','public_holiday_flag','local_event_flag','category','shelf_life_days','volume_cm3']]).fillna(0)
feature_cols = [c for c in feat.columns if c not in ['timestamp','quantity_sold']]
feat = feat.sort_values('timestamp')
split_idx = int(len(feat)*0.8)
train = feat.iloc[:split_idx].copy()
test = feat.iloc[split_idx:].copy()
X_train, y_train = train[feature_cols], train['quantity_sold']
X_test, y_test = test[feature_cols], test['quantity_sold']
models = {
    'RandomForest': RandomForestRegressor(n_estimators=250, random_state=42, max_depth=10),
    'XGBoost': XGBRegressor(n_estimators=250, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=250, learning_rate=0.05, random_state=42)
}
results, preds, importances = [], {}, {}
for name, model in models.items():
    model.fit(X_train, y_train)
    p = model.predict(X_test)
    preds[name] = p
    m = metrics(y_test, p)
    m['Model'] = name
    results.append(m)
    importances[name] = pd.DataFrame({'feature': X_train.columns, 'importance': getattr(model, 'feature_importances_', np.zeros(len(X_train.columns)))}).sort_values('importance', ascending=False).head(15)
if PROPHET_OK:
    prophet_df = df.groupby('timestamp', as_index=False)['quantity_sold'].sum().rename(columns={'timestamp':'ds','quantity_sold':'y'})
    prop_split = int(len(prophet_df)*0.8)
    prop_train = prophet_df.iloc[:prop_split]
    prop_test = prophet_df.iloc[prop_split:]
    m = Prophet(daily_seasonality=True, weekly_seasonality=True)
    m.fit(prop_train)
    forecast = m.predict(prop_test[['ds']])
    p = forecast['yhat'].values
    preds['Prophet'] = p
    pm = metrics(prop_test['y'].values, p)
    pm['Model'] = 'Prophet'
    results.append(pm)
else:
    st.warning('Prophet is not installed in this environment. Tree-based models will still run.')
res_df = pd.DataFrame(results)
st.dataframe(res_df, use_container_width=True)
st.plotly_chart(px.bar(res_df.melt(id_vars='Model', var_name='Metric', value_name='Value'), x='Model', y='Value', color='Metric', barmode='group', title='Model evaluation metrics'), use_container_width=True)
for name, p in preds.items():
    if name == 'Prophet' and PROPHET_OK:
        actual = prop_test['y'].values
        idx = prop_test['ds']
    else:
        actual = y_test.values
        idx = test['timestamp']
    comp = pd.DataFrame({'timestamp': idx, 'Actual': actual, 'Predicted': p})
    st.plotly_chart(px.line(comp, x='timestamp', y=['Actual','Predicted'], title=f'Actual vs Predicted - {name}'), use_container_width=True)
for name, fi in importances.items():
    st.plotly_chart(px.bar(fi.sort_values('importance'), x='importance', y='feature', orientation='h', title=f'Feature importance - {name}'), use_container_width=True)
st.subheader('Business actions')
st.markdown('- Increase safety stock for evening peak hours and for rain-sensitive staples.
- Raise reorder points for high-velocity dairy and bakery SKUs in stores with elevated stockout rates.
- Use flash discounts for short shelf-life items when rolling demand weakens versus trailing averages.
- Build weather-triggered replenishment playbooks for cola, ice cream, milk, and bread.')
