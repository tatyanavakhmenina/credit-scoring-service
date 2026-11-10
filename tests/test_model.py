import numpy as np
import pandas as pd

from credit_scoring.ml.features import add_credit_features
from credit_scoring.ml.model import ModelInfo, ScoringModel


def test_features_are_finite_when_bills_are_zero(features):
    features |= {f"bill_amt{i}": 0 for i in range(1, 7)}

    result = add_credit_features(pd.DataFrame([features]))

    assert np.isfinite(result[["utilization", "payment_ratio"]].to_numpy()).all()
    assert result["months_delayed"].iloc[0] == 2
    assert result["max_delay"].iloc[0] == 2


def test_scoring_model_applies_threshold_from_registry(features):
    class Pipeline:
        def predict_proba(self, frame):
            assert list(frame.columns)[0] == "limit_bal"
            return np.array([[0.7, 0.3]])

    model = ScoringModel(Pipeline(), ModelInfo("m", "champion", "1", "r", 0.25))

    probability = model.predict_proba(features)

    assert probability == 0.3
    assert model.decide(probability) == "reject"
