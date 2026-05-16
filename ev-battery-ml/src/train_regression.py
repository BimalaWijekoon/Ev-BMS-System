"""
Training script for regression models — CLI version of Notebook 04.
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

def load_and_split():
    df = pd.read_csv(DATA_PATH).drop(columns=['timestamp']).drop_duplicates()
    n = len(df)
    t_end, v_end = int(n*0.70), int(n*0.85)
    targets = ['cycle_degradation','over_temp_flag','over_voltage_flag']
    splits = []
    for s in [df.iloc[:t_end], df.iloc[t_end:v_end], df.iloc[v_end:]]:
        splits.append((s.drop(columns=targets), s['cycle_degradation']))
    return splits

def train_models():
    print("="*60+"\nRegression Training\n"+"="*60)
    (X_tr,y_tr),(X_v,y_v),(X_te,y_te) = load_and_split()
    pp = BatteryPreprocessor()
    Xtr = pp.fit_transform(X_tr); Xv = pp.transform(X_v); Xte = pp.transform(X_te)
    pp.save(MODELS_DIR)
    ytr_l, yv_l, yte_l = np.log1p(y_tr), np.log1p(y_v), np.log1p(y_te)
    results = {}; tscv = TimeSeriesSplit(n_splits=5)

    # Linear Regression baseline
    t0=time.time(); lr=LinearRegression(); lr.fit(Xtr,ytr_l); lt=time.time()-t0
    yp = np.expm1(lr.predict(Xv))
    results['LinearRegression'] = {'MAE':float(mean_absolute_error(y_v,yp)),'RMSE':float(np.sqrt(mean_squared_error(y_v,yp))),'R2':float(r2_score(y_v,yp)),'Time':round(lt,3)}
    print(f"LR: MAE={results['LinearRegression']['MAE']:.8f} RMSE={results['LinearRegression']['RMSE']:.8f} R2={results['LinearRegression']['R2']:.4f}")

    # Random Forest
    t0=time.time()
    gs_rf = GridSearchCV(RandomForestRegressor(random_state=42),
        {'n_estimators':[100,200],'max_depth':[5,10,None],'min_samples_split':[2,5],'max_features':['sqrt',0.5]},
        cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
    gs_rf.fit(Xtr,ytr_l); rf=gs_rf.best_estimator_; rt=time.time()-t0
    yp = np.expm1(rf.predict(Xv))
    results['RandomForest'] = {'MAE':float(mean_absolute_error(y_v,yp)),'RMSE':float(np.sqrt(mean_squared_error(y_v,yp))),'R2':float(r2_score(y_v,yp)),'Time':round(rt,3)}
    print(f"RF: MAE={results['RandomForest']['MAE']:.8f} RMSE={results['RandomForest']['RMSE']:.8f} R2={results['RandomForest']['R2']:.4f}")

    # RF feature importance plot
    imp = rf.feature_importances_; idx = np.argsort(imp)[::-1]; fn = pp.feature_columns
    fig,ax=plt.subplots(figsize=(12,6)); ax.barh(range(len(idx)),imp[idx][::-1])
    ax.set_yticks(range(len(idx))); ax.set_yticklabels([fn[i] for i in idx][::-1])
    ax.set_xlabel('Importance'); ax.set_title('RF Feature Importances')
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR,'rf_feature_importance.png'),dpi=150,bbox_inches='tight'); plt.close()

    # XGBoost
    t0=time.time()
    gs_xgb = GridSearchCV(XGBRegressor(objective='reg:squarederror',random_state=42,eval_metric='rmse',verbosity=0),
        {'n_estimators':[200,500],'max_depth':[3,5,7],'learning_rate':[0.01,0.05,0.1],'reg_alpha':[0,1.0],'reg_lambda':[1,10]},
        cv=tscv, scoring='neg_root_mean_squared_error', n_jobs=-1)
    gs_xgb.fit(Xtr,ytr_l)
    xgb = XGBRegressor(**gs_xgb.best_params_,objective='reg:squarederror',random_state=42,eval_metric='rmse',early_stopping_rounds=50,verbosity=0)
    xgb.fit(Xtr,ytr_l,eval_set=[(Xtr,ytr_l),(Xv,yv_l)],verbose=False); xt=time.time()-t0
    yp = np.expm1(xgb.predict(Xv))
    results['XGBoost'] = {'MAE':float(mean_absolute_error(y_v,yp)),'RMSE':float(np.sqrt(mean_squared_error(y_v,yp))),'R2':float(r2_score(y_v,yp)),'Time':round(xt,3)}
    print(f"XGB: MAE={results['XGBoost']['MAE']:.8f} RMSE={results['XGBoost']['RMSE']:.8f} R2={results['XGBoost']['R2']:.4f}")

    # XGB training curve
    ev = xgb.evals_result()
    fig,ax=plt.subplots(figsize=(10,5)); ax.plot(ev['validation_0']['rmse'],label='Train'); ax.plot(ev['validation_1']['rmse'],label='Val')
    ax.set_xlabel('Round'); ax.set_ylabel('RMSE'); ax.set_title('XGBoost Training Curve'); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR,'xgb_training_curve_regression.png'),dpi=150,bbox_inches='tight'); plt.close()

    # Select best
    best_name = min(results, key=lambda k: results[k]['RMSE'])
    best = {'LinearRegression':lr,'RandomForest':rf,'XGBoost':xgb}[best_name]
    joblib.dump(best, os.path.join(MODELS_DIR,'regression_model.pkl'))
    with open(os.path.join(MODELS_DIR,'regression_results.json'),'w') as f: json.dump(results,f,indent=2)
    print(f"\nBEST: {best_name}")

    # Learning curve
    ts,trs,vs = learning_curve(best,Xtr,ytr_l,train_sizes=np.linspace(0.1,1.0,10),cv=tscv,scoring='neg_root_mean_squared_error',n_jobs=-1)
    fig,ax=plt.subplots(figsize=(10,6)); ax.plot(ts,-trs.mean(1),'o-',label='Train'); ax.plot(ts,-vs.mean(1),'o-',label='Val')
    ax.set_xlabel('Size'); ax.set_ylabel('RMSE'); ax.set_title(f'Learning Curve — {best_name}'); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR,'learning_curve_regression.png'),dpi=150,bbox_inches='tight'); plt.close()
    return results

if __name__ == '__main__':
    train_models()
