from config import load_cfg
from slowfast.models import SlowFast

cfg = load_cfg('./test_config.yaml')
model = SlowFast(cfg)