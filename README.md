# StockAI

An end-to-end ML-driven trading research toolkit. It fetches market data, engineers features, trains an ensemble of models (RandomForest, GradientBoosting, XGBoost, LightGBM), generates signals, backtests with risk management, and reports metrics and plots.

## Features
- Data fetching via Yahoo Finance
- Robust feature engineering (duplicate-column safe)
- PCA + ensemble modeling
- Confidence-based signal generation
- Backtesting with risk control and trailing stops
- Walk-forward validation
- Comprehensive performance metrics and plots

## Project structure
```
core.py
main.py
plot.py
config/
  config.yaml
data/
  output/
    logs/
src/
  data/
    data_fetcher.py
    data_preprocessor.py
    feature_engineer.py
  models/
    model_trainer.py
    model_evaluator.py
  trading/
    backtester.py
    risk_manager.py
    signal_generator.py
  utils/
    config_loader.py
    imports.py
    logger.py
    metrics.py
```

Core pipeline functions:
- Fetch: [`src.data.data_fetcher.fetch_stock_data`](src/data/data_fetcher.py)
- Basic features: [`src.data.data_preprocessor.create_basic_features`](src/data/data_preprocessor.py)
- Prepare dataset: [`src.data.feature_engineer.prepare_data`](src/data/feature_engineer.py)
- Train: [`src.models.model_trainer.train_models`](src/models/model_trainer.py)
- Evaluate: [`src.models.model_evaluator.evaluate_models`](src/models/model_evaluator.py)
- Signals: [`src.trading.signal_generator.generate_signals`](src/trading/signal_generator.py)
- Backtest: [`src.trading.backtester.enhanced_simulate_trading`](src/trading/backtester.py)
- Metrics: [`src.utils.metrics.calculate_enhanced_metrics`](src/utils/metrics.py)
- Entry point: [main.py](main.py)

## Setup
- Python 3.10+
- Create a virtual environment and install deps:
  - Windows PowerShell
    ```
    py -m venv venv
    venv\Scripts\activate
    pip install -r requirements.txt
    ```
  - macOS/Linux
    ```
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```

## Configuration
Edit [config/config.yaml](config/config.yaml):
- data.symbol: Ticker (e.g., "AAPL")
- data.years: Lookback in years
- model.lookback_days: Rolling windows for features
- model.confidence_threshold: 0.5–0.9
- model.validation:
  - "backtest": single split
  - "walk_forward": rolling out-of-sample windows
- trading: risk parameters and costs

## Usage
- Backtest:
  ```
  python main.py --config config/config.yaml
  ```
- Walk-forward:
  - Set model.validation: "walk_forward" in [config/config.yaml](config/config.yaml)
  ```
  python main.py --config config/config.yaml
  ```

Outputs (data/output/):
- {SYMBOL}_model_evaluation.csv
- {SYMBOL}_trading_results.csv
- {SYMBOL}_trade_history.csv
- {SYMBOL}_performance_metrics.csv
- {SYMBOL}_enhanced_results.png
- {SYMBOL}_walk_forward_results.csv
- logs/trading_bot.log



## License
MIT — see [LICENSE](LICENSE).