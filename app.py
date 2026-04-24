import os
import io
import json
import warnings
import numpy as np
import pandas as pd
import subprocess
import sys

warnings.filterwarnings('ignore')

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse

# ── Auto-install ML deps if missing ───────────────────────────────────────────
for pkg in ['xgboost', 'lightgbm', 'scikit-learn']:
    try:
        __import__(pkg.replace('-', '_'))
    except ImportError:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

from sklearn.linear_model import Lasso, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb
import lightgbm as lgb

app = FastAPI(title="House Price Predictor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(train_df: pd.DataFrame, test_df: pd.DataFrame):
    y = np.log1p(train_df['SalePrice'])
    test_ids = test_df['Id'].copy()

    train_df = train_df.drop(columns=['Id', 'SalePrice'], errors='ignore')
    test_df  = test_df.drop(columns=['Id'], errors='ignore')
    n_train  = len(train_df)

    all_data = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    all_data['MSSubClass'] = all_data['MSSubClass'].astype(str)

    # Ordinal encoding
    qual_map = {'None':0,'Po':1,'Fa':2,'TA':3,'Gd':4,'Ex':5}
    for c in ['ExterQual','ExterCond','BsmtQual','BsmtCond','HeatingQC',
              'KitchenQual','FireplaceQu','GarageQual','GarageCond','PoolQC']:
        all_data[c] = all_data[c].map(qual_map).fillna(0).astype(int)

    all_data['BsmtExposure'] = all_data['BsmtExposure'].map(
        {'None':0,'No':1,'Mn':2,'Av':3,'Gd':4}).fillna(0).astype(int)
    fin = {'None':0,'Unf':1,'LwQ':2,'Rec':3,'BLQ':4,'ALQ':5,'GLQ':6}
    all_data['BsmtFinType1'] = all_data['BsmtFinType1'].map(fin).fillna(0).astype(int)
    all_data['BsmtFinType2'] = all_data['BsmtFinType2'].map(fin).fillna(0).astype(int)
    all_data['Functional']   = all_data['Functional'].map(
        {'Sal':1,'Sev':2,'Maj2':3,'Maj1':4,'Mod':5,'Min2':6,'Min1':7,'Typ':8}).fillna(5).astype(int)
    all_data['GarageFinish'] = all_data['GarageFinish'].map(
        {'None':0,'Unf':1,'RFn':2,'Fin':3}).fillna(0).astype(int)
    all_data['PavedDrive']   = all_data['PavedDrive'].map({'N':0,'P':1,'Y':2}).fillna(0).astype(int)
    all_data['LotShape']     = all_data['LotShape'].map(
        {'IR3':1,'IR2':2,'IR1':3,'Reg':4}).fillna(4).astype(int)
    all_data['LandSlope']    = all_data['LandSlope'].map(
        {'Sev':1,'Mod':2,'Gtl':3}).fillna(3).astype(int)
    all_data['CentralAir']   = all_data['CentralAir'].map({'N':0,'Y':1}).fillna(0).astype(int)

    # Missing values
    for c in ['PoolQC','MiscFeature','Alley','Fence','GarageType','MasVnrType']:
        if c in all_data.columns:
            all_data[c] = all_data[c].fillna('None')

    for c in ['GarageYrBlt','GarageArea','GarageCars','BsmtFinSF1','BsmtFinSF2',
              'BsmtUnfSF','TotalBsmtSF','BsmtFullBath','BsmtHalfBath','MasVnrArea']:
        if c in all_data.columns:
            all_data[c] = all_data[c].fillna(0)

    if 'Neighborhood' in all_data.columns and 'LotFrontage' in all_data.columns:
        all_data['LotFrontage'] = all_data.groupby('Neighborhood')['LotFrontage'].transform(
            lambda x: x.fillna(x.median()))

    for c in all_data.select_dtypes(include='object').columns:
        all_data[c].fillna(all_data[c].mode()[0], inplace=True)
    for c in all_data.select_dtypes(include=[np.number]).columns:
        all_data[c].fillna(all_data[c].median(), inplace=True)

    # Feature engineering
    all_data['TotalSF']      = all_data.get('TotalBsmtSF',0) + all_data.get('1stFlrSF',0) + all_data.get('2ndFlrSF',0)
    all_data['TotalBath']    = (all_data.get('FullBath',0) + 0.5*all_data.get('HalfBath',0) +
                                all_data.get('BsmtFullBath',0) + 0.5*all_data.get('BsmtHalfBath',0))
    all_data['TotalPorchSF'] = (all_data.get('OpenPorchSF',0) + all_data.get('EnclosedPorch',0) +
                                all_data.get('3SsnPorch',0) + all_data.get('ScreenPorch',0) +
                                all_data.get('WoodDeckSF',0))
    all_data['HouseAge']     = all_data.get('YrSold',0) - all_data.get('YearBuilt',0)
    all_data['RemodAge']     = all_data.get('YrSold',0) - all_data.get('YearRemodAdd',0)
    all_data['HasPool']      = (all_data.get('PoolArea',0) > 0).astype(int)
    all_data['HasGarage']    = (all_data.get('GarageArea',0) > 0).astype(int)
    all_data['HasBsmt']      = (all_data.get('TotalBsmtSF',0) > 0).astype(int)
    all_data['HasFireplace'] = (all_data.get('Fireplaces',0) > 0).astype(int)
    all_data['IsNew']        = (all_data.get('YearBuilt',0) == all_data.get('YrSold',0)).astype(int)
    all_data['WasRemodeled'] = (all_data.get('YearRemodAdd',0) != all_data.get('YearBuilt',0)).astype(int)
    all_data['QualSF']       = all_data.get('OverallQual',0) * all_data['TotalSF']
    all_data['QualLivArea']  = all_data.get('OverallQual',0) * all_data.get('GrLivArea',0)
    all_data['OverallScore'] = all_data.get('OverallQual',0) * all_data.get('OverallCond',0)

    # Skew correction
    num_feats = all_data.select_dtypes(include=[np.number]).columns
    skewed    = all_data[num_feats].apply(lambda x: x.skew()).abs()
    for c in skewed[skewed > 0.75].index:
        all_data[c] = np.log1p(all_data[c].clip(lower=0))

    all_data = pd.get_dummies(all_data)
    all_data.fillna(all_data.median(numeric_only=True), inplace=True)

    X_train = all_data.iloc[:n_train, :].values
    X_test  = all_data.iloc[n_train:, :].values
    return X_train, X_test, y, test_ids


def train_and_predict(X_train, X_test, y):
    lasso = make_pipeline(RobustScaler(), Lasso(alpha=0.0005, max_iter=10000, random_state=42))
    enet  = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=0.9, max_iter=10000, random_state=42))
    gbr   = GradientBoostingRegressor(n_estimators=1500, learning_rate=0.05, max_depth=4,
                                       max_features='sqrt', min_samples_leaf=15,
                                       min_samples_split=10, loss='huber', random_state=42)
    xgbm  = xgb.XGBRegressor(colsample_bytree=0.46, gamma=0.047, learning_rate=0.05,
                               max_depth=3, min_child_weight=1.78, n_estimators=1500,
                               reg_alpha=0.46, reg_lambda=0.86, subsample=0.52,
                               random_state=42, n_jobs=-1, verbosity=0)
    lgbm  = lgb.LGBMRegressor(objective='regression', num_leaves=5, learning_rate=0.05,
                                n_estimators=720, max_bin=55, bagging_fraction=0.8,
                                bagging_freq=5, feature_fraction=0.23,
                                min_data_in_leaf=6, min_sum_hessian_in_leaf=11,
                                random_state=42, verbose=-1, n_jobs=-1)

    models = [('Lasso', lasso), ('ElasticNet', enet), ('GBR', gbr),
              ('XGBoost', xgbm), ('LightGBM', lgbm)]

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    test_preds, weights, cv_scores = [], [], {}

    for name, model in models:
        scores = np.sqrt(-cross_val_score(model, X_train, y,
                          scoring='neg_mean_squared_error', cv=kf))
        cv_scores[name] = round(float(scores.mean()), 5)
        model.fit(X_train, y)
        pred = model.predict(X_test)
        test_preds.append(pred)
        weights.append(1.0 / scores.mean())

    weights     = np.array(weights) / sum(weights)
    final_log   = np.average(test_preds, axis=0, weights=weights)
    final_price = np.expm1(final_log)
    return final_price, cv_scores


@app.get("/")
def root():
    return {"message": "House Price Predictor API is running!"}


@app.post("/predict")
async def predict(train_file: UploadFile = File(...), test_file: UploadFile = File(...)):
    try:
        train_bytes = await train_file.read()
        test_bytes  = await test_file.read()
        train_df    = pd.read_csv(io.BytesIO(train_bytes))
        test_df     = pd.read_csv(io.BytesIO(test_bytes))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read CSV files: {str(e)}")

    if 'SalePrice' not in train_df.columns:
        raise HTTPException(status_code=400, detail="train.csv must contain a 'SalePrice' column")
    if 'Id' not in test_df.columns:
        raise HTTPException(status_code=400, detail="test.csv must contain an 'Id' column")

    try:
        X_train, X_test, y, test_ids = preprocess(train_df, test_df)
        predictions, cv_scores       = train_and_predict(X_train, X_test, y)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model error: {str(e)}")

    results = [{"Id": int(id_), "SalePrice": round(float(p), 2)}
               for id_, p in zip(test_ids, predictions)]

    return JSONResponse({
        "status": "success",
        "total_predictions": len(results),
        "cv_scores": cv_scores,
        "stats": {
            "min":  round(float(predictions.min()), 2),
            "max":  round(float(predictions.max()), 2),
            "mean": round(float(predictions.mean()), 2),
        },
        "predictions": results
    })


@app.post("/predict/csv")
async def predict_csv(train_file: UploadFile = File(...), test_file: UploadFile = File(...)):
    try:
        train_df = pd.read_csv(io.BytesIO(await train_file.read()))
        test_df  = pd.read_csv(io.BytesIO(await test_file.read()))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    X_train, X_test, y, test_ids = preprocess(train_df, test_df)
    predictions, _               = train_and_predict(X_train, X_test, y)

    out = io.StringIO()
    pd.DataFrame({"Id": test_ids.values, "SalePrice": predictions}).to_csv(out, index=False)
    out.seek(0)

    return StreamingResponse(
        io.BytesIO(out.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=predictions.csv"}
    )
