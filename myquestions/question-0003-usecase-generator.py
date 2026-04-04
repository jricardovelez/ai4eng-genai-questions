import random
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

def generar_caso_de_uso_evaluar_impacto_escalado_knn():
    """
    Genera un caso de uso aleatorio (input/output esperado)
    para evaluar_impacto_escalado_knn(X, y, n_neighbors, test_size, random_state)
    """
    rng = np.random.default_rng()

    n_samples = random.randint(150, 400)
    n_features = random.randint(4, 12)

    # Features con escalas MUY diferentes (para que el escalado importe)
    X = np.empty((n_samples, n_features))
    for j in range(n_features):
        scale = 10 ** rng.uniform(0, 4)  # escalas entre 1 y 10000
        X[:, j] = rng.normal(loc=rng.uniform(-10, 10), scale=scale, size=n_samples)

    # Clasificación basada en features normalizados (para que escalar ayude)
    X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
    w = rng.normal(size=n_features)
    logits = X_norm @ w + rng.normal(scale=0.5, size=n_samples)

    n_classes = random.choice([2, 3])
    if n_classes == 2:
        thr = np.median(logits)
        y = (logits > thr).astype(int)
    else:
        thresholds = np.quantile(logits, [1/3, 2/3])
        y = np.digitize(logits, thresholds)

    n_neighbors = random.choice([3, 5, 7, 9, 11])
    test_size = round(random.choice([0.2, 0.25, 0.3]), 2)
    random_state = random.randint(0, 50)

    input_data = {
        "X": X.copy(),
        "y": y.copy(),
        "n_neighbors": n_neighbors,
        "test_size": test_size,
        "random_state": random_state,
    }

    # ---- Ground truth ----
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Sin escalado
    knn_raw = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_raw.fit(X_train, y_train)
    pred_raw = knn_raw.predict(X_test)
    acc_raw = accuracy_score(y_test, pred_raw)
    f1_raw = f1_score(y_test, pred_raw, average="weighted", zero_division=0)

    # Con escalado
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    knn_scaled = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn_scaled.fit(X_train_s, y_train)
    pred_scaled = knn_scaled.predict(X_test_s)
    acc_scaled = accuracy_score(y_test, pred_scaled)
    f1_scaled = f1_score(y_test, pred_scaled, average="weighted", zero_division=0)

    output_data = {
        "acc_sin_escalar": round(float(acc_raw), 6),
        "f1_sin_escalar": round(float(f1_raw), 6),
        "acc_con_escalar": round(float(acc_scaled), 6),
        "f1_con_escalar": round(float(f1_scaled), 6),
        "mejora_accuracy": round(float(acc_scaled - acc_raw), 6),
        "mejora_f1": round(float(f1_scaled - f1_raw), 6),
    }

    return input_data, output_data
