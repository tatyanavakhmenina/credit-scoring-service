async def test_process_uses_loaded_model(client, state, features):
    response = await client.post("/process", json=features)

    assert response.status_code == 200
    data = response.json()
    assert data["probability_default"] == 0.9
    assert data["decision"] == "reject"
    assert data["model"]["version"] == "7"
    assert data["model"]["alias"] == "champion"


async def test_process_without_model_returns_503(client, state, features):
    state.model = None

    response = await client.post("/process", json=features)

    assert response.status_code == 503


async def test_process_rejects_invalid_features(client, state, features):
    features["age"] = 5

    response = await client.post("/process", json=features)

    assert response.status_code == 422


async def test_process_rejects_unknown_field(client, state, features):
    features["sex"] = 1

    response = await client.post("/process", json=features)

    assert response.status_code == 422
