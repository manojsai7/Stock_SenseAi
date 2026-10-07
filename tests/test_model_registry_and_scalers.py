import pytest
from backend.services.model_registry import model_registry

def test_supported_stocks_list():
    expected = ["TCS.NS", "RELIANCE.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS", "ATHERENERG.NS"]
    for sym in expected:
        assert model_registry.is_symbol_supported(sym)
    assert not model_registry.is_symbol_supported("UNKNOWN.NS")

def test_model_artifacts_exist_for_all_five_stocks():
    for sym in model_registry.trained_symbols:
        assert model_registry.has_artifacts(sym), f"Missing artifacts for {sym}"

def test_model_and_scaler_loading():
    # Test loading RELIANCE.NS
    model = model_registry.get_model("RELIANCE.NS")
    scaler = model_registry.get_scaler("RELIANCE.NS")
    meta = model_registry.get_metadata("RELIANCE.NS")

    assert model is not None
    assert scaler is not None
    assert scaler.is_fitted
    assert meta["symbol"] == "RELIANCE.NS"
    assert meta["window_size"] == 30
    assert len(meta["features"]) == 6

def test_unsupported_stock_raises():
    with pytest.raises(FileNotFoundError):
        model_registry.load_stock_artifacts("NONEXISTENT.NS")
