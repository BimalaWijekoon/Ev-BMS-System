"""
Training script for regression models — CLI version of Notebook 04.
Target: internal_resistance (Ω) — the primary electrochemical aging indicator.
Uses TimeSeriesSplit for cross-validation to prevent data leakage.
Usage: python src/train_regression.py
"""
import json, os, sys, time, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, learning_curve
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'nev_battery_charging.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
PLOTS_DIR = os.path.join(BASE_DIR, 'plots')
sys.path.insert(0, BASE_DIR)
from src.preprocessor import BatteryPreprocessor

REGRESSION_TARGET = 'internal_resistance'

def load_and_split():
    df = pd.read_csv(DATA_PATH).drop(columns=['timestamp']).drop_duplicates()
    n = len(df)
    t_end, v_end = int(n*0.70), int(n*0.85)
    targets = ['cycle_degradation','over_temp_flag','over_voltage_flag', 'internal_resistance']
    splits = []
    for s in [df.iloc[:t_end], df.iloc[t_end:v_end], df.iloc[v_end:]]:
        X = s.drop(columns=[c for c in targets if c in s.columns])
        y = s[REGRESSION_TARGET]
        splits.append((X, y))
    return splits

def train_models():
    print("="*60+f"\nRegression Training — Target: {REGRESSION_TARGET} (Ω)\n"+"="*60)
    (X_tr,y_tr),(X_v,y_v),(X_te,y_te) = load_and_split()
    pp = BatteryPreprocessor()
    Xtr = pp.fit_transform(X_tr); Xv = pp.transform(X_v); Xte = pp.transform(X_te)
    pp.save(MODELS_DIR)

    # No log transform needed — IR values (0.019–0.150 Ω) are already clean
    results = {}; tscv = TimeSeriesSplit(n_splits=5)

    # Linear Regression baseline
    t0=time.time(); lr=LinearRegression(); lr.fit(Xtr,y_tr); lt=time.time()-t0
    yp = lr.predict(Xv)
    results['LinearRegression'] = {'MAE':float(mean_absolute_error(y_v,yp)),'RMSE':float(np.sqrt(mean_squared_error(y_v,yp))),'R2':float(r2_score(y_v,yp)),'Time':round(lt,3)}
    print(f"LR: MAE={results['LinearRegression']['MAE']:.6f} Ω  RMSE={results['LinearRegression']['RMSE']:.6f} Ω  R²={results['LinearRegression']['R2']:.4f}")

    # Random Forest
    t0=time.time()
    gs_rf = GridSearchCV(RandomForestRegressor(random_state=42),
        {'n_estimators':[100,200],'max_depth':[5,10,None],'min_samples_split':[2,5],'max_features':['sqrt',0.5]},
        cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
    gs_rf.fit(Xtr,y_tr); rf=gs_rf.best_estimator_; rt=time.time()-t0
    yp = rf.predict(Xv)
    results['RandomForest'] = {'MAE':float(mean_absolute_error(y_v,yp)),'RMSE':float(np.sqrt(mean_squared_error(y_v,yp))),'R2':float(r2_score(y_v,yp)),'Time':round(rt,3)}
    print(f"RF: MAE={results['RandomForest']['MAE']:.6f} Ω  RMSE={results['RandomForest']['RMSE']:.6f} Ω  R²={results['RandomForest']['R2']:.4f}")

    # RF feature importance plot
    imp = rf.feature_importances_; idx = np.argsort(imp)[::-1]; fn = pp.feature_columns
    fig,ax=plt.subplots(figsize=(12,6)); ax.barh(range(len(idx)),imp[idx][::-1])
    ax.set_yticks(range(len(idx))); ax.set_yticklabels([fn[i] for i in idx][::-1])
    ax.set_xlabel('Importance'); ax.set_title('RF Feature Importances — Internal Resistance')
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR,'rf_feature_importance.png'),dpi=150,bbox_inches='tight'); plt.close()

    # XGBoost
    t0=time.time()
    gs_xgb = GridSearchCV(XGBRegressor(objective='reg:squarederror',random_state=42,eval_metric='rmse',verbosity=0),
        {'n_estimators':[200,500],'max_depth':[3,5,7],'learning_rate':[0.01,0.05,0.1],'reg_alpha':[0,1.0],'reg_lambda':[1,10]},
        cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
    gs_xgb.fit(Xtr,y_tr)
    xgb = XGBRegressor(**gs_xgb.best_params_,objective='reg:squarederror',random_state=42,eval_metric='rmse',early_stopping_rounds=50,verbosity=0)
    xgb.fit(Xtr,y_tr,eval_set=[(Xtr,y_tr),(Xv,y_v)],verbose=False); xt=time.time()-t0
    yp = xgb.predict(Xv)
    results['XGBoost'] = {'MAE':float(mean_absolute_error(y_v,yp)),'RMSE':float(np.sqrt(mean_squared_error(y_v,yp))),'R2':float(r2_score(y_v,yp)),'Time':round(xt,3)}
    print(f"XGB: MAE={results['XGBoost']['MAE']:.6f} Ω  RMSE={results['XGBoost']['RMSE']:.6f} Ω  R²={results['XGBoost']['R2']:.4f}")

    # XGB training curve
    ev = xgb.evals_result()
    fig,ax=plt.subplots(figsize=(10,5)); ax.plot(ev['validation_0']['rmse'],label='Train'); ax.plot(ev['validation_1']['rmse'],label='Val')
    ax.set_xlabel('Round'); ax.set_ylabel('RMSE (Ω)'); ax.set_title('XGBoost Training Curve — Internal Resistance'); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR,'xgb_training_curve_regression.png'),dpi=150,bbox_inches='tight'); plt.close()

    # Select best
    best_name = min(results, key=lambda k: results[k]['RMSE'])
    best = {'LinearRegression':lr,'RandomForest':rf,'XGBoost':xgb}[best_name]
    joblib.dump(best, os.path.join(MODELS_DIR,'regression_model.pkl'))
    with open(os.path.join(MODELS_DIR,'regression_results.json'),'w') as f: json.dump(results,f,indent=2)
    print(f"\nBEST: {best_name}")

    # Learning curve — use clean model without early_stopping for sklearn compatibility
    if best_name == 'XGBoost':
        lc_model = XGBRegressor(**gs_xgb.best_params_, objective='reg:squarederror', random_state=42, verbosity=0)
    else:
        lc_model = best
    ts,trs,vs = learning_curve(lc_model,Xtr,y_tr,train_sizes=np.linspace(0.1,1.0,10),cv=tscv,scoring='neg_root_mean_squared_error',n_jobs=-1)
    fig,ax=plt.subplots(figsize=(10,6)); ax.plot(ts,-trs.mean(1),'o-',label='Train'); ax.plot(ts,-vs.mean(1),'o-',label='Val')
    ax.set_xlabel('Size'); ax.set_ylabel('RMSE (Ω)'); ax.set_title(f'Learning Curve — {best_name}'); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR,'learning_curve_regression.png'),dpi=150,bbox_inches='tight'); plt.close()
    return results

if __name__ == '__main__':
    train_models()
