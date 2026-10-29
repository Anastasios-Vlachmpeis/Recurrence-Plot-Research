"""Published Table V values transcribed from the source paper."""
import pandas as pd

from .config import STOCKS

# Each token is mean:reported-plus-minus, in STOCKS order followed by printed Avg.
PUBLISHED = {
    'ARIMA': {
        'accuracy': '52.20:1.71 50.44:4.26 43.36:2.58 53.85:3.70 48.67:2.82 52.21:2.65 53.98:3.15 50.68:4.62',
        'f1': '.52:.01 .53:.02 .42:.01 .50:.03 .53:.01 .51:.01 .54:.02 .51:.04',
        'auc': '.54:.01 .50:.00 .51:.06 .53:.03 .47:.02 .54:.01 .51:.02 .51:.04'},
    'RF': {
        'accuracy': '49.56:2.15 53.9:.04 52.21:1.45 56.73:2.35 51.33:2.17 49.56:2.50 51.33:.71 52.10:3.13',
        'f1': '.48:.03 .54:.00 .53:.02 .55:.02 .57:.04 .46:.04 .58:.02 .53:.05',
        'auc': '.54:.03 .51:.03 .49:.03 .52:.03 .51:.03 .52:.02 .51:.02 .51:.03'},
    'XGBoost': {
        'accuracy': '47.79:1.45 51.33:2.10 48.67:.42 49.04:.72 51.33:.72 50.44:2.83 53.98:2.25 50.37:2.63',
        'f1': '.55:.01 .58:.03 .56:.02 .56:.07 .58:.02 .55:.02 .60:.06 .57:.04',
        'auc': '.52:.02 .49:.01 .53:.01 .52:.01 .46:.01 .49:.01 .51:.01 .50:.01'},
    'LSTM': {
        'accuracy': '54.69:3.18 60.94:1.92 57.03:1.27 55.36:1.36 46.09:2.41 57.03:2.49 58.59:3.76 55.68:5.02',
        'f1': '.48:.04 .59:.04 .62:.06 .57:.03 .52:.02 .62:.06 .56:.01 .57:.06',
        'auc': '.48:.05 .51:.02 .52:.05 .55:.04 .48:.04 .50:.06 .54:.01 .51:.05'},
    'Non-stationary Transformer': {
        'accuracy': '57.81:2.00 65.18:3.46 58.59:1.92 65.18:.91 64.06:.91 48.44:2.36 65.62:1.27 60.70:6.18',
        'f1': '.52:.03 .59:.02 .53:.03 .64:.02 .69:.04 .40:.05 .69:.04 .58:.10',
        'auc': '.45:.01 .50:.04 .49:.05 .60:.01 .58:.01 .49:.02 .56:.03 .53:.06'},
    'DLinear': {
        'accuracy': '47.66:1.28 62.50:1.27 49.22:2.56 58.93:.91 60.94:4.56 49.22:1.72 64.06:1.27 56.08:6.95',
        'f1': '.35:.04 .63:.01 .36:.03 .56:.02 .58:.05 .54:.01 .69:.04 .53:.12',
        'auc': '.51:.01 .53:.01 .46:.02 .57:.01 .55:.00 .48:.02 .53:.01 .52:.06'},
    'Dual-close': {
        'accuracy': '59.38:.97 60.94:1.61 64.06:2.41 59.82:2.34 65.62:1.91 64.84:.97 63.28:1.47 62.56:2.25',
        'f1': '.64:.01 .66:.01 .63:.02 .68:.03 .67:.02 .55:.01 .71:.02 .65:.06',
        'auc': '.52:.01 .55:.02 .55:.02 .56:.02 .57:.02 .54:.01 .56:.01 .55:.02'}}


def paper_reference():
    rows = []
    for model, measures in PUBLISHED.items():
        for metric, encoded in measures.items():
            tokens = encoded.split()
            if len(tokens) != 8:
                raise ValueError(f'Expected seven stock values and one average for {model}/{metric}')
            for stock, token in zip([*STOCKS, 'Average'], tokens):
                mean, sd = map(float, token.split(':'))
                factor = 100 if metric == 'accuracy' else 1
                rows.append(dict(stock=stock, model=model, metric=metric,
                                 published=mean / factor, published_reported_sd=sd / factor))
    return pd.DataFrame(rows)


PAPER_REFERENCE = paper_reference()
