import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def generar_caso_de_uso_seleccion_features_varianza():
    """
    Genera un caso de uso aleatorio (input/output esperado)
    para seleccion_features_varianza(X, y, threshold, test_size, random_state)
    """
    rng = np.random.default_rng()

    n_samples = random.randint(150, 400)
    n_informative = random.randint(4, 8)
    n_low_var = random.randint(3, 7)
    n_features = n_informative + n_low_var

    # Features informativos con varianza normal
    X_info = rng.normal(loc=0, scale=rng.uniform(1, 5, size=n_informative), size=(n_samples, n_informative))

    # Features de baja varianza (casi constantes)
    X_low = np.full((n_samples, n_low_var), 0.0)
    for j in range(n_low_var):
        base_val = rng.uniform(-5, 5)
        noise_scale = rng.uniform(0.001, 0.1)
        X_low[:, j] = base_val + rng.normal(0, noise_scale, size=n_samples)

    X = np.hstack([X_info, X_low])

    # Mezclar columnas
    perm = rng.permutation(n_features)
    X = X[:, perm]

    # Generar y basado en features informativos
    w = rng.normal(size=n_informative)
    logits = X_info @ w + rng.normal(scale=0.5, size=n_samples)
    y = (logits > np.median(logits)).astype(int)

    # Threshold que elimine los de baja varianza
    variances_low = [rng.uniform(0.001, 0.1) ** 2 for _ in range(n_low_var)]
    threshold = round(float(rng.uniform(0.05, 0.5)), 3)

    test_size = round(random.choice([0.2, 0.25, 0.3]), 2)
    random_state = random.randint(0, 50)

    input_data = {
        "X": X.copy(),
        "y": y.copy(),
        "threshold": threshold,
        "test_size": test_size,
        "random_state": random_state,
    }

    # ---- Ground truth ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # VarianceThreshold
    vt = VarianceThreshold(threshold=threshold)
    vt.fit(X_train)
    X_train_filt = vt.transform(X_train)
    X_test_filt = vt.transform(X_test)

    # Si quedan 0 features, fallback a threshold=0
    if X_train_filt.shape[1] == 0:
        vt = VarianceThreshold(threshold=0.0)
        vt.fit(X_train)
        X_train_filt = vt.transform(X_train)
        X_test_filt = vt.transform(X_test)

    # Modelo con todos los features
    dt_full = DecisionTreeClassifier(random_state=random_state)
    dt_full.fit(X_train, y_train)
    pred_full = dt_full.predict(X_test)
    acc_full = accuracy_score(y_test, pred_full)

    # Modelo con features filtrados
    dt_filt = DecisionTreeClassifier(random_state=random_state)
    dt_filt.fit(X_train_filt, y_train)
    pred_filt = dt_filt.predict(X_test_filt)
    acc_filt = accuracy_score(y_test, pred_filt)

    output_data = {
        "n_features_original": int(X.shape[1]),
        "n_features_filtrado": int(X_train_filt.shape[1]),
        "features_eliminados": int(X.shape[1] - X_train_filt.shape[1]),
        "acc_original": round(float(acc_full), 6),
        "acc_filtrado": round(float(acc_filt), 6),
        "diferencia_acc": round(float(acc_filt - acc_full), 6),
    }

    return input_data, output_data
