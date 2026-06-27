import warnings, json
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
try:
    from prophet import Prophet
    PROPHET_OK = True
except Exception:
    PROPHET_OK = False
outdir = 'output/qcommerce_demand_forecasting'
df = pd.read_csv(f'{outdir}/synthetic_qcommerce_demand.csv', parse_dates=['timestamp'])
sales_hour = df.groupby(df['timestamp'].dt.hour)['quantity_sold'].mean()
sales_day = df.groupby(df['timestamp'].dt.day_name())['quantity_sold'].mean().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday'])
weather = df.groupby('weather_condition')['quantity_sold'].agg(['mean','sum','count'])
holiday = df.groupby('public_holiday_flag')['quantity_sold'].mean()
stockout_by_store = df.groupby('store_id')['stockout_flag'].mean().sort_values(ascending=False)
def make_features(df):
    data = df.copy().sort_values(['store_id','sku_id','timestamp'])
    data['hour'] = data['timestamp'].dt.hour
    data['dayofweek'] = data['timestamp'].dt.dayofweek
    data['day'] = data['timestamp'].dt.day
    data['weekofyear'] = data['timestamp'].dt.isocalendar().week.astype(int)
    data['month'] = data['timestamp'].dt.month
    data['is_weekend'] = (data['dayofweek']>=5).astype(int)
    for lag in [1,2,24]:
        data[f'lag_{lag}'] = data.groupby(['store_id','sku_id'])['quantity_sold'].shift(lag)
    for win in [3,6,12,24]:
        data[f'roll_mean_{win}'] = data.groupby(['store_id','sku_id'])['quantity_sold'].transform(lambda s: s.shift(1).rolling(win, min_periods=1).mean())
    data = pd.get_dummies(data, columns=['store_id','sku_id','category','weather_condition'], drop_first=False)
    return data.fillna(0)
def metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    denom = np.where(np.array(y_true)==0, 1, np.array(y_true))
    mape = float(np.mean(np.abs((np.array(y_true)-np.array(y_pred))/denom))*100)
    r2 = float(r2_score(y_true, y_pred))
    return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}
feat = make_features(df[['timestamp','store_id','sku_id','quantity_sold','price_at_sale','stockout_flag','weather_condition','temperature','public_holiday_flag','local_event_flag','category','shelf_life_days','volume_cm3']]).sort_values('timestamp')
feature_cols = [c for c in feat.columns if c not in ['timestamp','quantity_sold']]
split_idx = int(len(feat)*0.8)
train = feat.iloc[:split_idx]
test = feat.iloc[split_idx:]
X_train, y_train = train[feature_cols], train['quantity_sold']
X_test, y_test = test[feature_cols], test['quantity_sold']
models = {
    'RandomForest': RandomForestRegressor(n_estimators=250, random_state=42, max_depth=10),
    'XGBoost': XGBRegressor(n_estimators=250, learning_rate=0.05, max_depth=6, subsample=0.9, colsample_bytree=0.9, random_state=42),
    'LightGBM': LGBMRegressor(n_estimators=250, learning_rate=0.05, random_state=42)
}
results=[]
preds={}
feat_imp={}
for name, model in models.items():
    model.fit(X_train, y_train)
    p = model.predict(X_test)
    preds[name]=p
    m = metrics(y_test, p)
    m['Model']=name
    results.append(m)
    feat_imp[name] = pd.DataFrame({'feature': X_train.columns, 'importance': model.feature_importances_}).sort_values('importance', ascending=False).head(15)
if PROPHET_OK:
    prophet_df = df.groupby('timestamp', as_index=False)['quantity_sold'].sum().rename(columns={'timestamp':'ds','quantity_sold':'y'})
    prop_split = int(len(prophet_df)*0.8)
    prop_train = prophet_df.iloc[:prop_split]
    prop_test = prophet_df.iloc[prop_split:]
    model = Prophet(daily_seasonality=True, weekly_seasonality=True)
    model.fit(prop_train)
    forecast = model.predict(prop_test[['ds']])
    p = forecast['yhat'].values
    preds['Prophet']=p
    m = metrics(prop_test['y'].values, p)
    m['Model']='Prophet'
    results.append(m)
res_df = pd.DataFrame(results)
res_df.to_csv(f'{outdir}/model_metrics.csv', index=False)
plt.figure(figsize=(10,5))
for metric in ['RMSE','MAE','MAPE','R2']:
    plt.plot(res_df['Model'], res_df[metric], marker='o', label=metric)
plt.legend(); plt.title('Model Evaluation Metrics'); plt.tight_layout(); plt.savefig(f'{outdir}/metrics_plot.png', dpi=160); plt.close()
for name, p in preds.items():
    plt.figure(figsize=(11,4))
    if name=='Prophet' and PROPHET_OK:
        actual = prop_test['y'].values
        x = pd.to_datetime(prop_test['ds'])
    else:
        actual = y_test.values
        x = pd.to_datetime(test['timestamp'])
    plt.plot(x, actual, label='Actual', linewidth=1.5)
    plt.plot(x, p, label='Predicted', linewidth=1.5)
    plt.title(f'Actual vs Predicted - {name}')
    plt.legend(); plt.tight_layout(); plt.savefig(f'{outdir}/actual_vs_pred_{name}.png', dpi=160); plt.close()
for name, fi in feat_imp.items():
    plt.figure(figsize=(8,5))
    fi = fi.sort_values('importance')
    plt.barh(fi['feature'], fi['importance'])
    plt.title(f'Feature Importance - {name}')
    plt.tight_layout(); plt.savefig(f'{outdir}/feature_importance_{name}.png', dpi=160); plt.close()
summary = {
    'rows': int(len(df)),
    'date_min': str(df['timestamp'].min()),
    'date_max': str(df['timestamp'].max()),
    'avg_qty_by_hour_top3': sales_hour.sort_values(ascending=False).head(3).round(2).to_dict(),
    'avg_qty_by_day': sales_day.round(2).to_dict(),
    'weather_mean_qty': weather['mean'].round(2).to_dict(),
    'holiday_mean_qty': holiday.round(2).to_dict(),
    'stockout_by_store': stockout_by_store.round(3).to_dict(),
    'best_model_by_rmse': res_df.sort_values('RMSE').iloc[0]['Model']
}
with open(f'{outdir}/analysis_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)
