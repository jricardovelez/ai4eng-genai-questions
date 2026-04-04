import random
import numpy as np
import pandas as pd

def generar_caso_de_uso_calcular_rfm():
    """
    Genera un caso de uso aleatorio (input/output esperado)
    para calcular_rfm(df, customer_col, date_col, amount_col, ref_date)
    """
    rng = np.random.default_rng()

    n_customers = random.randint(20, 60)
    customers = [f"cliente_{i:03d}" for i in range(n_customers)]

    # Generar transacciones
    n_transactions = random.randint(150, 500)
    rows = []
    base_date = pd.Timestamp("2024-01-01")
    date_range_days = random.randint(180, 365)

    for _ in range(n_transactions):
        cust = rng.choice(customers)
        days_offset = int(rng.integers(0, date_range_days))
        date = base_date + pd.Timedelta(days=days_offset)
        amount = round(float(rng.uniform(5, 500)), 2)
        rows.append((cust, date, amount))

    df = pd.DataFrame(rows, columns=["cliente", "fecha_compra", "monto"])

    # Inyectar datos sucios
    df["fecha_compra"] = df["fecha_compra"].astype(object)
    n_bad = max(2, n_transactions // 30)
    for idx in rng.choice(df.index.to_numpy(), size=n_bad, replace=False):
        df.loc[idx, "fecha_compra"] = rng.choice([None, "invalida", "2024-99-99"])

    df["monto"] = df["monto"].astype(object)
    n_bad_amount = max(1, n_transactions // 40)
    for idx in rng.choice(df.index.to_numpy(), size=n_bad_amount, replace=False):
        df.loc[idx, "monto"] = rng.choice(["no_num", None])

    n_bad_cust = max(1, n_transactions // 50)
    for idx in rng.choice(df.index.to_numpy(), size=n_bad_cust, replace=False):
        df.loc[idx, "cliente"] = np.nan

    ref_date = base_date + pd.Timedelta(days=date_range_days + random.randint(1, 30))
    ref_date_str = ref_date.strftime("%Y-%m-%d")

    input_data = {
        "df": df.copy(),
        "customer_col": "cliente",
        "date_col": "fecha_compra",
        "amount_col": "monto",
        "ref_date": ref_date_str,
    }

    # ---- Ground truth ----
    dfx = df.copy()
    dfx["fecha_compra"] = pd.to_datetime(dfx["fecha_compra"], errors="coerce")
    dfx["monto"] = pd.to_numeric(dfx["monto"], errors="coerce")
    dfx = dfx.dropna(subset=["fecha_compra", "monto", "cliente"]).copy()

    ref_dt = pd.to_datetime(ref_date_str)

    rfm = dfx.groupby("cliente").agg(
        recency_days=("fecha_compra", lambda x: (ref_dt - x.max()).days),
        frequency=("fecha_compra", "count"),
        monetary=("monto", "sum"),
    ).reset_index()

    rfm.columns = ["customer", "recency_days", "frequency", "monetary"]
    rfm["recency_days"] = rfm["recency_days"].astype(int)
    rfm["frequency"] = rfm["frequency"].astype(int)
    rfm["monetary"] = rfm["monetary"].round(2)

    # Calcular scores con qcut
    def safe_qcut(series, q, labels, ascending=True):
        try:
            if ascending:
                return pd.qcut(series, q=q, labels=labels, duplicates="drop").astype(float).fillna(1).astype(int)
            else:
                return pd.qcut(series, q=q, labels=labels, duplicates="drop").astype(float).fillna(1).astype(int)
        except Exception:
            return pd.Series([1] * len(series), index=series.index)

    rfm["r_score"] = safe_qcut(rfm["recency_days"], 4, [4, 3, 2, 1])
    rfm["f_score"] = safe_qcut(rfm["frequency"], 4, [1, 2, 3, 4])
    rfm["m_score"] = safe_qcut(rfm["monetary"], 4, [1, 2, 3, 4])

    rfm["rfm_score"] = rfm["r_score"] + rfm["f_score"] + rfm["m_score"]
    rfm["rfm_score"] = rfm["rfm_score"].astype(int)

    out = rfm[["customer", "recency_days", "frequency", "monetary", "rfm_score"]].copy()
    out = out.sort_values(["rfm_score", "customer"], ascending=[False, True]).reset_index(drop=True)

    output_data = out
    return input_data, output_data
