import random
import numpy as np
import pandas as pd

def generar_caso_de_uso_detectar_anomalias_rolling():
    """
    Genera un caso de uso aleatorio (input/output esperado)
    para detectar_anomalias_rolling(df, date_col, value_col, window, threshold)
    """
    rng = np.random.default_rng()

    n = random.randint(80, 250)
    base = pd.Timestamp("2023-01-01") + pd.Timedelta(days=random.randint(0, 300))
    freq_minutes = random.choice([5, 10, 15, 30, 60])

    dates = [base + pd.Timedelta(minutes=freq_minutes * i) for i in range(n)]
    # Serie con tendencia + ruido normal
    trend = np.linspace(0, rng.uniform(1, 5), n)
    noise = rng.normal(0, 1, n)
    values = trend + noise

    # Inyectar anomalías reales (picos grandes)
    n_spikes = random.randint(3, 10)
    spike_indices = rng.choice(range(n), size=min(n_spikes, n), replace=False)
    for idx in spike_indices:
        values[idx] += rng.choice([-1, 1]) * rng.uniform(5, 12)

    df = pd.DataFrame({"fecha": dates, "valor": values})

    # Inyectar fechas inválidas
    df["fecha"] = df["fecha"].astype(object)
    n_bad_dates = max(1, n // 40)
    for idx in rng.choice(df.index.to_numpy(), size=n_bad_dates, replace=False):
        df.loc[idx, "fecha"] = rng.choice(["no_es_fecha", None, "2023-13-45"])

    # Inyectar NaN en valores
    n_bad_vals = max(1, n // 50)
    for idx in rng.choice(df.index.to_numpy(), size=n_bad_vals, replace=False):
        df.loc[idx, "valor"] = np.nan

    window = random.choice([5, 7, 10, 15, 20])
    threshold = round(random.choice([1.5, 2.0, 2.5, 3.0]), 1)

    input_data = {
        "df": df.copy(),
        "date_col": "fecha",
        "value_col": "valor",
        "window": window,
        "threshold": threshold,
    }

    # ---- Ground truth ----
    dfx = df.copy()
    dfx["fecha"] = pd.to_datetime(dfx["fecha"], errors="coerce")
    dfx["valor"] = pd.to_numeric(dfx["valor"], errors="coerce")
    dfx = dfx.dropna(subset=["fecha", "valor"]).sort_values("fecha").reset_index(drop=True)

    rolling_mean = dfx["valor"].rolling(window=window, min_periods=window).mean()
    rolling_std = dfx["valor"].rolling(window=window, min_periods=window).std()

    z_score = (dfx["valor"] - rolling_mean) / rolling_std
    z_score = z_score.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    is_anomaly = z_score.abs() > threshold
    anomaly_indices = sorted(dfx.index[is_anomaly].tolist())

    total_rows = len(dfx)
    total_anomalies = int(is_anomaly.sum())
    anomaly_pct = round(total_anomalies / total_rows * 100, 6) if total_rows > 0 else 0.0

    if total_anomalies > 0:
        mean_abs_zscore = round(float(z_score[is_anomaly].abs().mean()), 6)
    else:
        mean_abs_zscore = 0.0

    output_data = {
        "total_rows": total_rows,
        "total_anomalies": total_anomalies,
        "anomaly_pct": anomaly_pct,
        "anomaly_indices": anomaly_indices,
        "mean_abs_zscore": mean_abs_zscore,
    }

    return input_data, output_data
