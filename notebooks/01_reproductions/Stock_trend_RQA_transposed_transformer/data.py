"""Tushare acquisition, integrity caches, sessions, labels and chronological splits."""
import os, time, json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from .config import sha_bytes, fingerprint, json_save
from .models import DualBranchClassifier
from .training import seed_everything
MISSING_TOKEN = ('TUSHARE_TOKEN is required to download missing historical minute data; '
                 'separate A-share historical-minute permission is required.')


class DataService:
    def __init__(self, config, rqa):
        self.config=config
        self.rqa=rqa
        self._last_request=0.0

    def cache_paths(self, endpoint, code, start, end):
        safe = lambda s: str(s).replace(':', '').replace(' ', '_')
        path = self.config.RAW / endpoint / f'{code}_{safe(start)}_{safe(end)}.csv.gz'
        return (path, path.with_suffix(path.suffix + '.json'))

    def cached_frame(self, endpoint, code, start, end, query):
        path, meta_path = self.cache_paths(endpoint, code, start, end)
        identity = dict(endpoint=endpoint, code=code, start=start, end=end, frequency='1min' if endpoint == 'stk_mins' else 'daily', adjustment='provider_raw_unadjusted')
        if path.exists() or meta_path.exists():
            if not path.exists() or not meta_path.exists():
                raise RuntimeError(f'Incomplete cache pair: {path}. Preserve/repair it before rerunning.')
            meta = json.loads(meta_path.read_text(encoding='utf8'))
            if any((meta.get(k) != v for k, v in identity.items())) or meta.get('sha256') != sha_bytes(path.read_bytes()):
                raise RuntimeError(f'Invalid cache metadata/hash: {path}; not overwritten.')
            df = pd.read_csv(path, dtype={'ts_code': str, 'trade_date': str, 'cal_date': str})
            if len(df) != meta.get('rows'):
                raise RuntimeError(f'Invalid cache row count: {path}')
            return df
        if query is None:
            raise RuntimeError(f'{MISSING_TOKEN} Missing: {endpoint} {code} {start} .. {end}. Calendar (2000 points) and daily coverage permissions are also required.')
        time.sleep(max(0.0, 1.3 - (time.monotonic() - self._last_request)))
        self._last_request = time.monotonic()
        try:
            df = query()
        except Exception as exc:
            raise RuntimeError(f'Tushare {endpoint} failed for {code} {start} .. {end}; check credentials, endpoint permission and connectivity ({type(exc).__name__}).') from None
        if not isinstance(df, pd.DataFrame):
            raise RuntimeError(f'{endpoint} returned no DataFrame for {code} {start} .. {end}')
        limit = 8000 if endpoint == 'stk_mins' else 6000 if endpoint == 'daily' else None
        if limit and len(df) >= limit:
            raise RuntimeError(f'Potentially truncated {endpoint} response for {code} {start} .. {end}')
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, compression='gzip')
        json_save(meta_path, {**identity, 'rows': len(df), 'sha256': sha_bytes(path.read_bytes()), 'retrieved_utc': datetime.now(timezone.utc).isoformat()})
        return df

    def load_stock_minutes(self, pro, ts_code, start, end):
        exchange = 'SSE' if ts_code.endswith('.SH') else 'SZSE'
        compact = lambda d: pd.Timestamp(d).strftime('%Y%m%d')
        cal = self.cached_frame('trade_cal', exchange, start, end, None if pro is None else lambda: pro.trade_cal(exchange=exchange, start_date=compact(start), end_date=compact(end), is_open='1'))
        daily = self.cached_frame('daily', ts_code, start, end, None if pro is None else lambda: pro.daily(ts_code=ts_code, start_date=compact(start), end_date=compact(end)))
        if not {'cal_date', 'is_open'}.issubset(cal) or not {'trade_date', 'ts_code'}.issubset(daily):
            raise ValueError(f'Missing calendar/daily coverage columns for {ts_code}')
        if not daily.ts_code.eq(ts_code).all():
            raise ValueError('Daily coverage returned another stock')
        frames = []
        for left in pd.date_range(start, end, freq='14D'):
            right = min(left + pd.Timedelta(days=13), pd.Timestamp(end))
            lo, hi = (left.strftime('%Y-%m-%d') + ' 00:00:00', right.strftime('%Y-%m-%d') + ' 23:59:59')
            expected = daily.trade_date.astype(str).between(compact(left), compact(right)).any()

            def fetch(lo=lo, hi=hi, expected=expected):
                data = pro.stk_mins(ts_code=ts_code, freq='1min', start_date=lo, end_date=hi)
                if data is None or (data.empty and expected):
                    raise ValueError('Minute data missing despite daily coverage')
                if data.empty and len(data.columns) == 0:
                    data = pd.DataFrame(columns=['ts_code', 'trade_time', *self.config.FEATURES, 'amount'])
                return data
            frame = self.cached_frame('stk_mins', ts_code, lo, hi, None if pro is None else fetch)
            if frame.empty and expected:
                raise RuntimeError(f'Missing cached minute data for {ts_code} {lo} .. {hi}')
            frames.append(frame)
        return (pd.concat(frames, ignore_index=True), cal, daily)

    def expected_slots(self, day):
        d = pd.Timestamp(day).strftime('%Y-%m-%d')
        return pd.date_range(d + ' 09:31', d + ' 11:30', freq='min', tz='Asia/Shanghai').append(pd.date_range(d + ' 13:01', d + ' 15:00', freq='min', tz='Asia/Shanghai'))

    def audit_and_build_samples(self, frame, calendar, daily, ts_code):
        required = {'ts_code', 'trade_time', *self.config.FEATURES}
        if not required.issubset(frame):
            raise ValueError(f'Missing minute columns: {sorted(required - set(frame))}')
        df = frame.copy()
        if not df.ts_code.eq(ts_code).all():
            raise ValueError('Minute response contains an unexpected stock')
        times = pd.to_datetime(df.pop('trade_time'), errors='raise')
        df.index = pd.DatetimeIndex(times).tz_localize('Asia/Shanghai') if times.dt.tz is None else pd.DatetimeIndex(times).tz_convert('Asia/Shanghai')
        df = df.sort_index()
        duplicate_count = int(df.index.duplicated().sum())
        if duplicate_count:
            dup = df.loc[df.index.duplicated(keep=False)]
            if (dup.groupby(level=0).nunique(dropna=False) > 1).any().any():
                raise ValueError('Conflicting duplicate minute timestamps')
            df = df.loc[~df.index.duplicated(keep='first')]
        vals = df[self.config.FEATURES].apply(pd.to_numeric, errors='raise')
        if not np.isfinite(vals.to_numpy()).all() or (vals[['open', 'close', 'high', 'low']] <= 0).any().any() or (vals.vol < 0).any():
            raise ValueError('Nonfinite/nonpositive prices or negative volume')
        if ((vals.low > vals[['open', 'close']].min(axis=1)) | (vals.high < vals[['open', 'close']].max(axis=1))).any():
            raise ValueError('Invalid OHLC relationships')
        cal = calendar.loc[calendar.is_open.astype(str).eq('1'), 'cal_date']
        days = pd.DatetimeIndex(pd.to_datetime(cal.astype(str))).normalize().sort_values().unique()
        if len(days) == 0:
            raise ValueError('No exchange trading days in calendar')
        covered = set(pd.to_datetime(daily.trade_date.astype(str)).dt.normalize())
        minutes = df.index.hour * 60 + df.index.minute
        regular = (minutes >= 571) & (minutes <= 690) | (minutes >= 781) & (minutes <= 900)
        regular &= (df.index.second == 0) & (df.index.microsecond == 0)
        day_keys = df.index.tz_localize(None).normalize()
        regular &= day_keys.isin(days)
        off_session = int((~regular).sum())
        vals = vals.loc[regular]
        valid = {}
        audit = []
        for d in days:
            rows = vals.loc[vals.index.tz_localize(None).normalize() == d]
            status = 'unavailable_or_suspended' if d not in covered else 'missing' if len(rows) == 0 else 'valid' if rows.index.equals(self.expected_slots(d)) else 'incomplete'
            audit.append(dict(stock=ts_code, date=d, year=d.year, status=status, bars=len(rows)))
            if status == 'valid':
                valid[d] = rows
        xs = []
        meta = []
        excluded = []
        for i in range(5, len(days)):
            span = days[i - 5:i + 1]
            if not all((d in valid for d in span)):
                excluded.append(days[i])
                continue
            x = pd.concat([valid[d] for d in span[:-1]])
            prev = float(valid[span[-2]].close.iloc[-1])
            target = float(valid[span[-1]].close.iloc[-1])
            if not x.index[-1].tz_localize(None).normalize() < span[-1]:
                raise AssertionError('Target-day information entered input')
            xs.append(x.to_numpy(dtype=np.float32))
            meta.append(dict(stock=ts_code, date=span[-1], input_start=x.index[0], input_end=x.index[-1], previous_close=prev, target_close=target, y=int(target >= prev)))
        metadata = pd.DataFrame(meta)
        x = np.stack(xs) if xs else np.empty((0, 1200, 5), dtype=np.float32)
        unique_rows = pd.concat(list(valid.values())).sort_index() if valid else vals.iloc[:0]
        print(ts_code, 'identical duplicates:', duplicate_count, 'off-session:', off_session, 'excluded target spans:', len(excluded))
        return dict(x=x, meta=metadata, minutes=unique_rows, audit=pd.DataFrame(audit), excluded_dates=pd.DatetimeIndex(excluded))

    def synthetic_minutes(self):
        """Explicit fixture, never called by historical full mode."""
        dates = pd.bdate_range('2018-02-01', periods=40).append(pd.bdate_range('2021-02-01', periods=24)).append(pd.bdate_range('2022-02-01', periods=24))
        rng = np.random.default_rng(904)
        frames = []
        last = 100.0
        for i, d in enumerate(dates):
            target = last + (0 if i == 10 else 0.6 if i % 2 else -0.5)
            close = np.linspace(last, target, 240) + 0.015 * np.sin(np.arange(240) * 0.19)
            close[-1] = target
            op = np.r_[last, close[:-1]]
            frames.append(pd.DataFrame(dict(ts_code='SYNTHETIC', trade_time=self.expected_slots(d).tz_localize(None), open=op, close=close, high=np.maximum(op, close) + 0.02, low=np.minimum(op, close) - 0.02, vol=rng.integers(100, 5000, 240), amount=close * 1000)))
            last = target
        return (pd.concat(frames, ignore_index=True), pd.DataFrame({'cal_date': dates.strftime('%Y%m%d'), 'is_open': '1'}), pd.DataFrame({'trade_date': dates.strftime('%Y%m%d'), 'ts_code': 'SYNTHETIC'}))

    def split_samples(self, data, smoke=False):
        meta = data['meta']
        years = meta.date.dt.year.to_numpy()
        fit = np.flatnonzero((years >= 2018) & (years <= 2020))
        val = np.flatnonzero(years == 2021)
        final = np.flatnonzero((years >= 2018) & (years <= 2021))
        test = np.flatnonzero(years == 2022)
        if not all((len(a) for a in [fit, val, final, test])):
            raise ValueError('A stock lacks fit, validation, final-fit or test samples')
        boundary = meta.input_start.iloc[val[0]].tz_localize(None).normalize()
        removed = fit[meta.date.iloc[fit].to_numpy() >= boundary.to_datetime64()]
        fit = np.setdiff1d(fit, removed)
        print('Purged selection targets:', meta.date.iloc[removed].dt.strftime('%Y-%m-%d').tolist())
        if smoke:
            fit, val, final, test = (fit[:12], val[:6], final[:12], test[:6])
        for name, idx in [('fit', fit), ('validation', val), ('final', final)]:
            if len(idx) == 0 or meta.y.iloc[idx].nunique() < 2:
                raise ValueError(f'{name} lacks samples or both classes')
        return dict(fit=fit, val=val, final=final, test=test)

    def missing_source_ranges(self):
        records = []

        def need(endpoint, code, lo, hi):
            path, meta = self.cache_paths(endpoint, code, lo, hi)
            if not (path.exists() and meta.exists()):
                records.append(dict(endpoint=endpoint, stock_or_exchange=code, start=lo, end=hi))
        for exchange in ['SSE', 'SZSE']:
            need('trade_cal', exchange, self.config.START, self.config.END)
        for stock in self.config.STOCKS:
            need('daily', stock, self.config.START, self.config.END)
            for left in pd.date_range(self.config.START, self.config.END, freq='14D'):
                right = min(left + pd.Timedelta(days=13), pd.Timestamp(self.config.END))
                need('stk_mins', stock, left.strftime('%Y-%m-%d') + ' 00:00:00', right.strftime('%Y-%m-%d') + ' 23:59:59')
        return pd.DataFrame(records)

    def acquire_datasets(self):
        if self.config.SMOKE:
            f, c, d = self.synthetic_minutes()
            print(self.config.LABEL, ': using deterministic fixtures; no network calls.')
            return {'SYNTHETIC': self.audit_and_build_samples(f, c, d, 'SYNTHETIC')}
        token = os.environ.get('TUSHARE_TOKEN')
        missing = self.missing_source_ranges()
        if not token and len(missing):
            self.config.RESULTS.mkdir(parents=True, exist_ok=True)
            missing.to_csv(self.config.RESULTS / 'missing_data.csv', index=False)
            print('Missing source requests:', len(missing))
            display(missing.head(12))
            raise RuntimeError(f'{MISSING_TOKEN} All missing source ranges: {self.config.RESULTS / 'missing_data.csv'}. Calendar (2000 points) and daily coverage endpoint permissions are also needed.')
        if token:
            import tushare as ts
            pro = ts.pro_api(token)
            f, c, d = self.load_stock_minutes(pro, '000151.SZ', '2018-01-02', '2018-01-12')
            probe = self.audit_and_build_samples(f, c, d, '000151.SZ')
            if len(probe['x']) == 0:
                raise RuntimeError('Historical probe produced no complete 1200-minute input')
            backend = self.rqa.choose_backend()
            self.rqa.check_fixture(backend)
            r = self.rqa.features(probe['x'][0, :, 1], backend)
            x = probe['x'][0]
            scale = x.std(0)
            scale[scale == 0] = 1
            seed_everything(0)
            model = DualBranchClassifier(self.config).eval()
            with torch.no_grad():
                out = model(torch.tensor(((x - x.mean(0)) / scale)[None]), torch.tensor(np.nan_to_num(r)[None]))
            if not torch.isfinite(out).all():
                raise RuntimeError('Nonfinite historical probe forward output')
            print('Real historical probe: complete session audit, 13 RQA values and model forward passed.')
        else:
            pro = None
        datasets = {}
        for stock in self.config.STOCKS:
            frame, cal, daily = self.load_stock_minutes(pro, stock, self.config.START, self.config.END)
            datasets[stock] = self.audit_and_build_samples(frame, cal, daily, stock)
        return datasets
