import json
import numbers
import os
import numpy as np
import yaml


class SafeFallbackJSONEncoder(json.JSONEncoder):

    def __init__(self, nan_str="null", **kwargs):
        super(SafeFallbackJSONEncoder, self).__init__(**kwargs)
        self.nan_str = nan_str

    def default(self, value):
        try:
            if np.isnan(value):
                return self.nan_str

            if (type(value).__module__ == np.__name__
                    and isinstance(value, np.ndarray)):
                return value.tolist()

            if issubclass(type(value), numbers.Integral):
                return int(value)
            if issubclass(type(value), numbers.Number):
                return float(value)

            return super(SafeFallbackJSONEncoder, self).default(value)

        except Exception:
            return str(value)  # give up, just stringify it (ok for logs)


def pretty_dict_str(result):
    result = result.copy()
    result.update(config=None)  # drop config from pretty print
    out = {}
    for k, v in result.items():
        if v is not None:
            out[k] = v

    cleaned = json.dumps(out, cls=SafeFallbackJSONEncoder)
    return yaml.safe_dump(json.loads(cleaned), default_flow_style=False)


def is_windows_os() -> bool:
    return os.name == 'nt'


def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
