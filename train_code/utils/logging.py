import os

def save_run_params(log_path, params: dict):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    def _write_dict(d, f, indent=0):
        for k, v in d.items():
            if isinstance(v, dict):
                f.write(" " * indent + f"{k}:\n")
                _write_dict(v, f, indent + 4)
            else:
                f.write(" " * indent + f"{k}: {v}\n")
    with open(log_path, "w") as f:
        _write_dict(params, f)