import yaml
from datetime import datetime, timedelta

def load_config(config_path='config/config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    if 'years' in config['data']:
        end_date = datetime.now().date()
        start_date = (datetime.now() - timedelta(days=365 * config['data']['years'])).date()
        config['data']['start_date'] = start_date
        config['data']['end_date'] = end_date

    config['model'].setdefault('lookback_days', [3, 5, 10, 20])
    config['model'].setdefault('confidence_threshold', 0.6)
    config['trading'].setdefault('max_portfolio_risk', 0.10)
    config['trading'].setdefault('stop_loss_pct', 0.03)
    
    return config