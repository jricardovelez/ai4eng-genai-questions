import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preparar_datos(df, target_col):
    y = df[target_col].values
    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    return X_scaled, y
