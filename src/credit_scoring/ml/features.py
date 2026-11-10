import numpy as np
import pandas as pd


def add_credit_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Загрузка кредитной линии и доля выплат: в сырых суммах этих отношений нет."""
    result = frame.copy()
    bills = [f"bill_amt{i}" for i in range(1, 7)]
    payments = [f"pay_amt{i}" for i in range(1, 7)]
    delays = ["pay_0", "pay_2", "pay_3", "pay_4", "pay_5", "pay_6"]
    total_bill = result[bills].sum(axis=1)
    result["utilization"] = result["bill_amt1"] / result["limit_bal"]
    result["payment_ratio"] = result[payments].sum(axis=1) / total_bill.clip(lower=1)
    result["months_delayed"] = (result[delays] > 0).sum(axis=1)
    result["max_delay"] = result[delays].max(axis=1)
    return result.replace([np.inf, -np.inf], 0.0)
