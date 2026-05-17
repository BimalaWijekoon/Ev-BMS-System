"""
Training script for classification models — CLI version of Notebook 05.
Trains classifiers for over_temp_flag with class imbalance handling.
Uses StratifiedKFold for CV (not TimeSeriesSplit) because over_temp_flag has a
temporal block pattern where chronological splits produce single-class folds.
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
    precision_score, recall_score, roc_auc_score, roc_curve, auc)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
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
    targets = ['cycle_degradation','over_temp_flag','over_voltage_flag','internal_resistance']
    splits = []
    for s in [df.iloc[:t_end], df.iloc[t_end:v_end], df.iloc[v_end:]]:
        X = s.drop(columns=[c for c in targets if c in s.columns])
        splits.append((X, s['over_temp_flag'], s['over_voltage_flag']))
    return splits

def train_classifiers():
    print("="*60+"\nClassification Training\n"+"="*60)
    (X_tr,yt_tr,yv_tr),(X_v,yt_v,yv_v),(X_te,yt_te,yv_te) = load_and_split()
    pp = BatteryPreprocessor.load(MODELS_DIR)
    Xtr = pp.transform(X_tr)
    results = {}

    # StratifiedKFold ensures both classes in every fold
    # Justified deviation from TimeSeriesSplit: over_temp_flag has block structure
    # (0→1 at ~row 900), making chronological CV folds single-class
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Over voltage flag analysis
    volt_pos = int(yv_tr.sum())
    print(f"\nover_voltage_flag: {volt_pos} positives in training ({volt_pos/len(yv_tr)*100:.2f}%)")
    if volt_pos < 20:
        print("  → Too few positives. Using rule-based fallback: action_voltage > 4.15 OR terminal_voltage > 4.18")

    # Class weights for over_temp
    yt_tr_arr = yt_tr.values
    cw = compute_class_weight('balanced', classes=np.array([0,1]), y=yt_tr_arr)
    neg, pos = (yt_tr_arr==0).sum(), (yt_tr_arr==1).sum()
    print(f"\nover_temp_flag: {pos} positives, {neg} negatives (ratio {neg/max(pos,1):.1f}:1)")

    # 1. Logistic Regression
    print("\n--- Logistic Regression ---")
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    lr_cv_probs = cross_val_predict(lr, Xtr, yt_tr_arr, cv=skf, method='predict_proba')[:,1]
    lr.fit(Xtr, yt_tr_arr)
    # Threshold tuning on CV predictions
    best_f1, best_th = 0, 0.5
    for th in [0.3,0.35,0.4,0.45,0.5]:
        yp = (lr_cv_probs>=th).astype(int)
        f = f1_score(yt_tr_arr, yp, zero_division=0)
        r = recall_score(yt_tr_arr, yp, zero_division=0)
        if f > best_f1 and r >= 0.80: best_f1, best_th = f, th
    yp_lr = (lr_cv_probs>=best_th).astype(int)
    results['LogisticRegression'] = {
        'Recall':float(recall_score(yt_tr_arr,yp_lr,zero_division=0)),
        'Precision':float(precision_score(yt_tr_arr,yp_lr,zero_division=0)),
        'F1':float(f1_score(yt_tr_arr,yp_lr,zero_division=0)),
        'ROC_AUC':float(roc_auc_score(yt_tr_arr,lr_cv_probs)),
        'Threshold':best_th}
    print(f"  Threshold={best_th} F1={results['LogisticRegression']['F1']:.4f} Recall={results['LogisticRegression']['Recall']:.4f}")

    # 2. SVM RBF
    print("\n--- SVM (RBF Kernel) ---")
    gs_svm = GridSearchCV(
        SVC(kernel='rbf', class_weight='balanced', probability=True, random_state=42),
        {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']},
        cv=skf, scoring='f1', n_jobs=-1)
    gs_svm.fit(Xtr, yt_tr_arr)
    best_svm = gs_svm.best_estimator_
    svm_cv_probs = cross_val_predict(best_svm, Xtr, yt_tr_arr, cv=skf, method='predict_proba')[:,1]
    yp_svm = (svm_cv_probs >= 0.5).astype(int)
    results['SVM_RBF'] = {
        'Recall':float(recall_score(yt_tr_arr,yp_svm,zero_division=0)),
        'Precision':float(precision_score(yt_tr_arr,yp_svm,zero_division=0)),
        'F1':float(f1_score(yt_tr_arr,yp_svm,zero_division=0)),
        'ROC_AUC':float(roc_auc_score(yt_tr_arr,svm_cv_probs)),
        'Threshold':0.5}
    print(f"  F1={results['SVM_RBF']['F1']:.4f} Recall={results['SVM_RBF']['Recall']:.4f}")

    # 3. XGBoost Classifier
    print("\n--- XGBoost Classifier ---")
    gs = GridSearchCV(
        XGBClassifier(scale_pos_weight=neg/max(pos,1), eval_metric='logloss', random_state=42, verbosity=0),
        {'n_estimators':[100,300],'max_depth':[3,5,7],'learning_rate':[0.01,0.05,0.1]},
        cv=skf, scoring='f1', n_jobs=-1)
    gs.fit(Xtr, yt_tr_arr)
    xgb = XGBClassifier(**gs.best_params_, scale_pos_weight=neg/max(pos,1),
        eval_metric='logloss', random_state=42, verbosity=0)
    xgb.fit(Xtr, yt_tr_arr)
    xgb_cv_probs = cross_val_predict(
        XGBClassifier(**gs.best_params_, scale_pos_weight=neg/max(pos,1),
            eval_metric='logloss', random_state=42, verbosity=0),
        Xtr, yt_tr_arr, cv=skf, method='predict_proba')[:,1]
    # Threshold tuning for recall >= 0.85
    best_f1x, best_thx = 0, 0.5
    for th in np.arange(0.2,0.7,0.05):
        yp = (xgb_cv_probs>=th).astype(int)
        f = f1_score(yt_tr_arr,yp,zero_division=0)
        r = recall_score(yt_tr_arr,yp,zero_division=0)
        if r >= 0.85 and f > best_f1x: best_f1x, best_thx = f, th
    if best_f1x == 0: best_thx = 0.5
    yp_xgb = (xgb_cv_probs>=best_thx).astype(int)
    results['XGBoost'] = {
        'Recall':float(recall_score(yt_tr_arr,yp_xgb,zero_division=0)),
        'Precision':float(precision_score(yt_tr_arr,yp_xgb,zero_division=0)),
        'F1':float(f1_score(yt_tr_arr,yp_xgb,zero_division=0)),
        'ROC_AUC':float(roc_auc_score(yt_tr_arr,xgb_cv_probs)),
        'Threshold':float(best_thx)}
    print(f"  Threshold={best_thx} F1={results['XGBoost']['F1']:.4f} Recall={results['XGBoost']['Recall']:.4f}")

    # ROC comparison plot (using CV predictions)
    fig,ax=plt.subplots(figsize=(8,6))
    for name, probs, color in [('LogReg',lr_cv_probs,'#2196F3'),('SVM-RBF',svm_cv_probs,'#4CAF50'),('XGBoost',xgb_cv_probs,'#FF9800')]:
        fpr,tpr,_ = roc_curve(yt_tr_arr, probs)
        ax.plot(fpr, tpr, color=color, linewidth=2, label=f'{name} (AUC={auc(fpr,tpr):.3f})')
    ax.plot([0,1],[0,1],'k--'); ax.set_title('ROC Curves — Over-Temperature (StratifiedKFold CV)')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR'); ax.legend()
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR,'roc_comparison_overtemp.png'),dpi=150,bbox_inches='tight'); plt.close()

    # Select best classifier (highest F1 with Recall >= 0.85)
    valid = {k:v for k,v in results.items() if v['Recall']>=0.85}
    if not valid: valid = results
    best_name = max(valid, key=lambda k: valid[k]['F1'])
    best_model = {'LogisticRegression':lr,'SVM_RBF':best_svm,'XGBoost':xgb}[best_name]
    best_threshold = results[best_name]['Threshold']
    joblib.dump(best_model, os.path.join(MODELS_DIR,'classification_model_temp.pkl'))
    config = {
        'temp_threshold':best_threshold,
        'voltage_rule':'action_voltage > 4.15 OR terminal_voltage > 4.18',
        'best_classifier': best_name,
        'cv_strategy': 'StratifiedKFold(n_splits=5, shuffle=True)',
        'cv_reason': 'over_temp_flag block structure makes TimeSeriesSplit folds single-class'
    }
    with open(os.path.join(MODELS_DIR,'classification_config.json'),'w') as f: json.dump(config,f,indent=2)
    with open(os.path.join(MODELS_DIR,'classification_results.json'),'w') as f: json.dump(results,f,indent=2)
    print(f"\nBEST CLASSIFIER: {best_name} (F1={results[best_name]['F1']:.4f})")
    return results

if __name__ == '__main__':
    train_classifiers()
