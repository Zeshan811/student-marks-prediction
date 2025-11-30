# model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.dummy import DummyRegressor

# ============================
# Function to evaluate models
# ============================
def evaluate_models(X, y, poly_degree=None, bootstrap=True, n_boot=500):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}
    models = {}

    # ---------- Linear Regression ----------
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    results['Linear'] = {
        'MAE': mean_absolute_error(y_test, y_pred_lr),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        'R2': r2_score(y_test, y_pred_lr),
        'Train_R2': r2_score(y_train, lr.predict(X_train))
    }
    models['Linear'] = {'model': lr, 'type': 'linear'}

    # ---------- Polynomial Regression ----------
    if poly_degree:
        poly = PolynomialFeatures(degree=poly_degree)
        X_train_poly = poly.fit_transform(X_train)
        X_test_poly = poly.transform(X_test)
        pr = LinearRegression()
        pr.fit(X_train_poly, y_train)
        y_pred_poly = pr.predict(X_test_poly)
        results[f'Polynomial_{poly_degree}'] = {
            'MAE': mean_absolute_error(y_test, y_pred_poly),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_poly)),
            'R2': r2_score(y_test, y_pred_poly),
            'Train_R2': r2_score(y_train, pr.predict(X_train_poly))
        }
        models[f'Polynomial_{poly_degree}'] = {'model': pr, 'type': 'poly', 'degree': poly_degree}

    # ---------- Dummy Regressor ----------
    dummy = DummyRegressor(strategy='mean')
    dummy.fit(X_train, y_train)
    y_dummy = dummy.predict(X_test)
    results['Dummy'] = {
        'MAE': mean_absolute_error(y_test, y_dummy),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_dummy)),
        'R2': r2_score(y_test, y_dummy),
        'Train_R2': r2_score(y_train, dummy.predict(X_train))
    }
    models['Dummy'] = {'model': dummy, 'type': 'dummy'}

    # ---------- Bootstrapping MAE 95% CI ----------
    if bootstrap:
        for name, info in models.items():
            model = info['model']
            if info['type'] == 'poly':
                X_tr = poly.fit_transform(X_train)
            else:
                X_tr = X_train
            y_tr = y_train
            X_te = X_test
            y_te = y_test
            mae_boot = []
            for _ in range(n_boot):
                idx = np.random.choice(len(X_tr), len(X_tr), replace=True)
                X_sample = X_tr[idx] if info['type']=='poly' else X_tr.iloc[idx]
                y_sample = y_tr.iloc[idx]
                model.fit(X_sample, y_sample)
                y_pred_boot = model.predict(X_te if info['type']!='poly' else poly.transform(X_te))
                mae_boot.append(mean_absolute_error(y_te, y_pred_boot))
            ci_lower = np.percentile(mae_boot, 2.5)
            ci_upper = np.percentile(mae_boot, 97.5)
            results[name]['MAE_95CI'] = (ci_lower, ci_upper)

    return results, models


# ============================
# Function to create comparison table
# ============================
def create_comparison_table(results):
    table = pd.DataFrame(results).T
    table['MAE_95CI'] = table['MAE_95CI'].apply(lambda x: f"{x[0]:.3f}-{x[1]:.3f}" if x else "")
    table = table[['MAE','RMSE','R2','Train_R2','MAE_95CI']]
    return table


# ============================
# Function to predict new marks
# ============================
def predict_marks(model_info, input_df):
    """
    model_info: dict containing 'model', 'type', and optionally 'degree'
    input_df: pd.DataFrame with correct features
    """
    model = model_info['model']
    if model_info['type'] == 'poly':
        poly = PolynomialFeatures(degree=model_info['degree'])
        input_df = poly.fit_transform(input_df)
    return model.predict(input_df)
