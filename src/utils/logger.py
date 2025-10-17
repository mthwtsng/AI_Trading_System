import os
import logging

def setup_logger():
    logger = logging.getLogger('TradingBot')
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.setLevel(logging.INFO)

    os.makedirs('data/output/logs', exist_ok=True)

    fh = logging.FileHandler('data/output/logs/trading_bot.log', mode='w')
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger