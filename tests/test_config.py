from train_code.config import Config

def test_config_properties():
    config = Config()
    assert isinstance(config.run_id, int)
    assert config.run_id > 0
    assert config.base_model_name.startswith("best_model_")
    assert config.best_model_path.endswith(".pth")
    assert config.metrics_path.endswith(".csv")
    assert config.plots_path.endswith(".png")
    assert "pruned" in config.pruned_model_path()
    assert "quantized" in config.quantized_model_path()
    assert config.tflite_model_path().endswith(".tflite")
    assert config.onnx_model_path().endswith(".onnx")