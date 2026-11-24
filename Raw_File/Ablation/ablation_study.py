import numpy as np
import pandas as pd
import json
import os
import sys
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from bagging import bagging_sampler

class MedicalEnsembleV3Ultimate:
    def __init__(self, models_to_use=None, random_state=42):
        self.random_state = random_state
        self.smote = SMOTE(random_state=self.random_state)
        
        all_models = {
            'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(penalty='l2', C=10, solver='lbfgs', random_state=random_state)),

            'Decision Tree': DecisionTreeClassifier(max_depth=15, min_samples_split=20, min_samples_leaf=10, criterion='gini', random_state=random_state),
            
            'SVM': make_pipeline(StandardScaler(), SVC(C=9239.1, gamma=0.041, kernel='rbf', probability=True, random_state=random_state))
        }
        
        if models_to_use is None:
            models_to_use = list(all_models.keys())
            
        self.model_names = models_to_use
        self.models = [all_models[name] for name in self.model_names]
        self.trained_models = []
        self.weights = []
        self.best_threshold = 0.5 

    def fit(self, bags, X_val, y_val, use_tuning=True):
        self.trained_models = []
        oob_f1_scores = []

        for model, name, bag in zip(self.models, self.model_names, bags):
            X_train_bag, y_train_bag = bag['train_X'], bag['train_y']
            X_oob, y_oob = bag['oob_X'], bag['oob_y']

            # Resample SMOTE
            X_smote, y_smote = self.smote.fit_resample(X_train_bag, y_train_bag)
            model.fit(X_smote, y_smote)
            self.trained_models.append(model)

            # OOB tuning
            oob_preds = model.predict(X_oob)
            f1 = f1_score(y_oob, oob_preds)
            oob_f1_scores.append(f1)

        total_f1 = sum(oob_f1_scores)
        if total_f1 == 0:
            self.weights = np.ones(len(self.trained_models)) / len(self.trained_models)
        else:
            self.weights = np.array(oob_f1_scores) / total_f1

        if use_tuning:
            val_probas = self._get_weighted_probas(X_val)
            thresholds = np.linspace(0.1, 0.9, 50)
            best_f1 = 0
            for t in thresholds:
                preds = (val_probas >= t).astype(int)
                f1 = f1_score(y_val, preds)
                if f1 > best_f1:
                    best_f1 = f1
                    self.best_threshold = t
        else:
            self.best_threshold = 0.5

    def _get_weighted_probas(self, X):
        probas = np.array([m.predict_proba(X)[:, 1] if hasattr(m, 'predict_proba') else m.decision_function(X) for m in self.trained_models])
        return np.average(probas, axis=0, weights=self.weights)

    def predict(self, X, voting='soft'):
        if voting == 'soft':
            weighted_probas = self._get_weighted_probas(X)
            return (weighted_probas >= self.best_threshold).astype(int)
        elif voting == 'hard':
            preds = np.array([m.predict(X) for m in self.trained_models])
            sum_votes = np.sum(preds, axis=0)
            # Voting đa số (>= 2 vote nếu có 3 mô hình, v.v...)
            threshold = len(self.trained_models) / 2.0
            return (sum_votes > threshold).astype(int)

    def predict_proba(self, X):
        return self._get_weighted_probas(X)

def run_experiment(name, bags, X_val, y_val, X_test, y_test, models=None, use_tuning=True, voting='soft'):
    ensemble = MedicalEnsembleV3Ultimate(models_to_use=models)
    ensemble.fit(bags, X_val, y_val, use_tuning=use_tuning)
    
    y_pred = ensemble.predict(X_test, voting=voting)
    y_prob = ensemble.predict_proba(X_test)
    
    weights_dict = dict(zip(ensemble.model_names, [float(w) for w in ensemble.weights])) if voting == 'soft' else "N/A (Hard Voting)"
    
    results = {
        'experiment': name,
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'auc': roc_auc_score(y_test, y_prob),
        'threshold': ensemble.best_threshold if voting == 'soft' else "N/A (Hard Voting)",
        'weights': weights_dict
    }
    
    print(f"Results: F1={results['f1']:.4f} | Precision={results['precision']:.4f} | Recall={results['recall']:.4f}")
    return results

if __name__ == "__main__":
    base_dir = r"f:\X-FILE\Code_UNI\Python\Machine Learning\CuoiKy\Tai"
    data_dir = os.path.join(base_dir, "processed_data")
    
    X_train = pd.read_csv(os.path.join(data_dir, "X_train_Dataset_2.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train_Dataset_2.csv")).iloc[:, -1]
    
    X_val = pd.read_csv(os.path.join(data_dir, "X_val_Dataset_2.csv"))
    y_val = pd.read_csv(os.path.join(data_dir, "y_val_Dataset_2.csv")).iloc[:, -1]
    
    X_test = pd.read_csv(os.path.join(data_dir, "X_test_Dataset_2.csv"))
    y_test = pd.read_csv(os.path.join(data_dir, "y_test_Dataset_2.csv")).iloc[:, -1]
    
    all_results = []
    
    # Full Ensemble
    bags_full = bagging_sampler(X_train, y_train, n_subsets=3, sample_size_ratio=0.3, random_state=42)
    all_results.append(run_experiment("Full Ensemble (Baseline v3 Ultimate)", bags_full, X_val, y_val, X_test, y_test))
    
    # Hard Voting
    all_results.append(run_experiment("Hard Voting (Majority)", bags_full, X_val, y_val, X_test, y_test, voting='hard'))
    
    # No Bagging (100% Data)
    bags_no_bagging = []
    for _ in range(3):
        bags_no_bagging.append({
            'train_X': X_train, 'train_y': y_train,
            'oob_X': X_val, 'oob_y': y_val 
        })
    all_results.append(run_experiment("No Bagging (100% Data)", bags_no_bagging, X_val, y_val, X_test, y_test))
    
    # No SVM (LR + DT)
    bags_2 = bagging_sampler(X_train, y_train, n_subsets=2, sample_size_ratio=0.3, random_state=42)
    all_results.append(run_experiment("No SVM (LR + DT)", bags_2, X_val, y_val, X_test, y_test, models=['Logistic Regression', 'Decision Tree']))
    
    # No Threshold Tuning (0.5 Default)
    all_results.append(run_experiment("No Threshold Tuning (0.5 Default)", bags_full, X_val, y_val, X_test, y_test, use_tuning=False))

    # Single Best Model (SVM Only)
    bags_1 = bagging_sampler(X_train, y_train, n_subsets=1, sample_size_ratio=1.0, random_state=42)
    all_results.append(run_experiment("Single Model (SVM Only)", bags_1, X_val, y_val, X_test, y_test, models=['SVM']))

    output_path = os.path.join(base_dir, "docs", "ablation", "results_v3_ultimate.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)
