"""Explicit experiment configuration and artifact identity helpers."""
from pathlib import Path
import os, sys, json, hashlib
from datetime import datetime, timezone
from importlib.metadata import version
import torch

STOCKS = ['000151.SZ','002407.SZ','002432.SZ','002564.SZ','002603.SZ','002864.SZ','600152.SH']
FEATURES = ['open','close','high','low','vol']

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def fingerprint(obj):
    return sha_bytes(json.dumps(obj, sort_keys=True, default=str, allow_nan=False).encode())

def json_save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str, allow_nan=False), encoding='utf8')

class ExperimentConfig:
    """Own paths, numerical settings and execution mode without import-time side effects."""
    def __init__(self, root=None, mode=None):
        cwd=Path.cwd().resolve() if root is None else Path(root).resolve()
        self.ROOT=next((p for p in [cwd,*cwd.parents] if (p/'src/recplay/embedding.py').is_file() and (p/'notebooks').is_dir()),None)
        if self.ROOT is None: raise RuntimeError('Run from RecPlay or one of its notebook directories.')
        if str(self.ROOT/'src') not in sys.path:sys.path.insert(0,str(self.ROOT/'src'))
        self.MODE=mode or os.environ.get('STOCK_REPRO_MODE','full')
        if self.MODE not in {'full','smoke'}:raise ValueError('STOCK_REPRO_MODE must be full or smoke')
        self.SMOKE=self.MODE=='smoke'
        self.LABEL='SMOKE / SYNTHETIC — NOT MARKET RESULTS' if self.SMOKE else 'Historical reconstruction'
        self.STOCKS=STOCKS.copy();self.FEATURES=FEATURES.copy()
        self.START,self.END='2018-01-01','2022-12-31'
        self.WINDOW_DAYS,self.BARS_PER_DAY,self.M,self.TAU,self.EPS_FRACTION=5,240,7,1,.10
        self.SEEDS=[0] if self.SMOKE else list(range(5))
        self.MAX_EPOCHS,self.PATIENCE=(2,2) if self.SMOKE else (100,15)
        self.DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
        self.RAW=self.ROOT/'data/raw/stock_trend_tushare'
        self.PROCESSED=self.ROOT/'data/processed/stock_trend_repro'/self.MODE
        self.RESULTS=self.ROOT/'results/tables/stock_trend_repro'/self.MODE
        self.VERSIONS={p:version(p) for p in ['numpy','pandas','scipy','scikit-learn','statsmodels','xgboost','tushare','PyRQA','torch','nbformat','nbclient']}
        self.CONFIG=dict(mode=self.MODE,stocks=self.STOCKS,start=self.START,end=self.END,features=self.FEATURES,
            window_days=5,bars_per_day=240,m=self.M,tau=self.TAU,epsilon_fraction=self.EPS_FRACTION,
            seeds=self.SEEDS,max_epochs=self.MAX_EPOCHS,patience=self.PATIENCE,lr=3e-6,
            weight_decay=.1,batch_size=16,implementation='su2025-modular-1')
        folder=Path(__file__).parent
        # Hash every implementation module as well as executable notebook cells.
        modules={p.name:sha_bytes(p.read_bytes()) for p in sorted(folder.glob('*.py'))}
        notebook=folder.with_suffix('.ipynb')
        document=json.loads(notebook.read_text(encoding='utf8'))
        cells=[''.join(c['source']) if isinstance(c['source'],list) else c['source'] for c in document['cells'] if c['cell_type']=='code']
        self.CONFIG['source_sha256']=fingerprint(dict(modules=modules,cells=cells))
        torch.backends.cudnn.benchmark=False
        torch.set_num_threads(min(4,os.cpu_count() or 1))
