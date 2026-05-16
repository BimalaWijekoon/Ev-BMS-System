"""
Training script for classification models — CLI version of Notebook 05.
Trains classifiers for over_temp_flag with class imbalance handling.
Uses rule-based fallback for over_voltage_flag due to extreme imbalance.
Usage: python src/train_classification.py
"""
import json, os, sys, time, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, RocCurveDisplay, PrecisionRecallDisplay)
from sklearn.utils.class_weight import compute_class_weight
from xgboost import XGBClassifier
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
        splits.append((s.drop(columns=targets), s['over_temp_flag'], s['over_voltage_flag']))
    return splits

def train_classifiers():
    print("="*60+"\nClassification Training\n"+"="*60)
    (X_tr,yt_tr,yv_tr),(X_v,yt_v,yv_v),(X_te,yt_te,yv_te) = load_and_split()
    pp = BatteryPreprocessor.load(MODELS_DIR)
    Xtr = pp.transform(X_tr); Xv = pp.transform(X_v)
    results = {}

    # Over voltage flag analysis
    volt_pos = int(yv_tr.sum())
    print(f"\nover_voltage_flag: {volt_pos} positives in training ({volt_pos/len(yv_tr)*100:.2f}%)")
    if volt_pos < 20:
        print("  → Too few positives. Using rule-based fallback: action_voltage > 4.15 OR terminal_voltage > 4.18")

    # Class weights for over_temp
    cw = compute_class_weight('balanced', classes=np.array([0,1]), y=yt_tr)
    class_weights = {0:cw[0], 1:cw[1]}
    neg, pos = (yt_tr==0).sum(), (yt_tr==1).sum()
    print(f"\nover_temp_flag: {pos} positives, {neg} negatives (ratio {neg/max(pos,1):.1f}:1)")

    # 1. Logistic Regression
    print("\n--- Logistic Regression ---")
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr.fit(Xtr, yt_tr)
    probs_lr = lr.predict_proba(Xv)[:,1]
    # Threshold tuning
    best_f1, best_th = 0, 0.5
    for th in [0.3,0.35,0.4,0.45,0.5]:
        yp = (probs_lr>=th).astype(int)
        f = f1_score(yt_v, yp); r = recall_score(yt_v, yp)
        if f > best_f1 and r >= 0.80: best_f1, best_th = f, th
    yp_lr = (probs_lr>=best_th).astype(int)
    results['LogisticRegression'] = {'Recall':float(recall_score(yt_v,yp_lr)),'Precision':float(precision_score(yt_v,yp_lr)),
        'F1':float(f1_score(yt_v,yp_lr)),'ROC_AUC':float(roc_auc_score(yt_v,probs_lr)),'Threshold':best_th}
    print(f"  Threshold={best_th} F1={results['LogisticRegression']['F1']:.4f} Recall={results['LogisticRegression']['Recall']:.4f}")

    # 2. SVM RBF
    print("\n--- SVM (RBF Kernel) ---")
    best_svm, best_svm_f1 = None, 0
    for C in [0.1,1,10]:
        for g in ['scale','auto']:
            svm = SVC(kernel='rbf',C=C,gamma=g,class_weight='balanced',probability=True,random_state=42)
            svm.fit(Xtr,yt_tr)
            f = f1_score(yt_v, svm.predict(Xv))
            if f > best_svm_f1: best_svm, best_svm_f1 = svm, f
    probs_svm = best_svm.predict_proba(Xv)[:,1]
    yp_svm = best_svm.predict(Xv)
    results['SVM_RBF'] = {'Recall':float(recall_score(yt_v,yp_svm)),'Precision':float(precision_score(yt_v,yp_svm)),
        'F1':float(f1_score(yt_v,yp_svm)),'ROC_AUC':float(roc_auc_score(yt_v,probs_svm)),'Threshold':0.5}
    print(f"  F1={results['SVM_RBF']['F1']:.4f} Recall={results['SVM_RBF']['Recall']:.4f}")

    # 3. XGBoost Classifier
    print("\n--- XGBoost Classifier ---")
    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    gs = GridSearchCV(XGBClassifier(scale_pos_weight=neg/max(pos,1),eval_metric='logloss',random_state=42,verbosity=0),
        {'n_estimators':[100,300],'max_depth':[3,5,7],'learning_rate':[0.01,0.05,0.1]},
        cv=tscv,scoring='f1',n_jobs=-1)
    gs.fit(Xtr,yt_tr)
    xgb = XGBClassifier(**gs.best_params_,scale_pos_weight=neg/max(pos,1),eval_metric='logloss',
        early_stopping_rounds=30,random_state=42,verbosity=0)
    xgb.fit(Xtr,yt_tr,eval_set=[(Xv,yt_v)],verbose=False)
    probs_xgb = xgb.predict_proba(Xv)[:,1]
    # Threshold tuning for recall >= 0.85
    best_f1x, best_thx = 0, 0.5
    for th in np.arange(0.2,0.7,0.05):
        yp = (probs_xgb>=th).astype(int)
        f, r = f1_score(yt_v,yp), recall_score(yt_v,yp)
        if r >= 0.85 and f > best_f1x: best_f1x, best_thx = f, th
    if best_f1x == 0: best_thx = 0.5  # fallback
    yp_xgb = (probs_xgb>=best_thx).astype(int)
    results['XGBoost'] = {'Recall':float(recall_score(yt_v,yp_xgb)),'Precision':float(precision_score(yt_v,yp_xgb)),
        'F1':float(f1_score(yt_v,yp_xgb)),'ROC_AUC':float(roc_auc_score(yt_v,probs_xgb)),'Threshold':float(best_thx)}
    print(f"  Threshold={best_thx} F1={results['XGBoost']['F1']:.4f} Recall={results['XGBoost']['Recall']:.4f}")

    # ROC comparison plot
    fig,ax=plt.subplots(figsize=(8,6))
    RocCurveDisplay.from_estimator(lr,Xv,yt_v,ax=ax,name='LogReg')
    RocCurveDisplay.from_estimator(best_svm,Xv,yt_v,ax=ax,name='SVM-RBF')
    RocCurveDisplay.from_estimator(xgb,Xv,yt_v,ax=ax,name='XGBoost')
    ax.plot([0,1],[0,1],'k--'); ax.set_title('ROC Curves — Over-Temperature Classification')
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR,'roc_comparison_overtemp.png'),dpi=150,bbox_inches='tight'); plt.close()

    # Select best classifier (highest F1 with Recall >= 0.85)
    valid = {k:v for k,v in results.items() if v['Recall']>=0.85}
    if not valid: valid = results
    best_name = max(valid, key=lambda k: valid[k]['F1'])
    best_model = {'LogisticRegression':lr,'SVM_RBF':best_svm,'XGBoost':xgb}[best_name]
    best_threshold = results[best_name]['Threshold']
    joblib.dump(best_model, os.path.join(MODELS_DIR,'classification_model_temp.pkl'))
    config = {'temp_threshold':best_threshold,'voltage_rule':'action_voltage > 4.15 OR terminal_voltage > 4.18'}
    with open(os.path.join(MODELS_DIR,'classification_config.json'),'w') as f: json.dump(config,f,indent=2)
    with open(os.path.join(MODELS_DIR,'classification_results.json'),'w') as f: json.dump(results,f,indent=2)
    print(f"\nBEST CLASSIFIER: {best_name} (F1={results[best_name]['F1']:.4f})")
    return results

if __name__ == '__main__':
    train_classifiers()
