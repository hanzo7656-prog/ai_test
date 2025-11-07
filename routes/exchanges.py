from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging
from complete_coinstats_manager import coin_stats_manager

logger = logging.getLogger(__name__)

exchanges_router = APIRouter(prefix="/api/exchanges", tags=["Exchanges"])

@exchanges_router.get("/list", summary="لیست صرافی‌ها")
async def get_exchanges_list():
    """دریافت لیست صرافی‌های پردازش شده"""
    try:
        raw_data = coin_stats_manager.get_tickers_exchanges()
        
        # پردازش داده‌های صرافی‌ها
        processed_exchanges = []
        for exchange in raw_data.get('data', []):
            processed_exchanges.append({
                'id': exchange.get('id'),
                'name': exchange.get('name'),
                'year_established': exchange.get('year_established'),
                'country': exchange.get('country'),
                'trust_score': exchange.get('trust_score'),
                'trade_volume_24h_btc': exchange.get('trade_volume_24h_btc'),
                'url': exchange.get('url'),
                'image': exchange.get('image'),
                'last_updated': datetime.now().isoformat()
            })
        
        return {
            'status': 'success',
            'data': processed_exchanges,
            'total': len(processed_exchanges),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in exchanges list: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@exchanges_router.get("/markets", summary="مارکت‌ها")
async def get_markets():
    """دریافت مارکت‌های پردازش شده"""
    try:
        raw_data = coin_stats_manager.get_markets()
        
        processed_markets = []
        for market in raw_data.get('result', []):
            processed_markets.append({
                'exchange_id': market.get('exchangeId'),
                'base_asset': market.get('baseAsset'),
                'quote_asset': market.get('quoteAsset'),
                'price': market.get('price'),
                'volume_24h': market.get('volume24h'),
                'last_updated': datetime.now().isoformat()
            })
        
        return {
            'status': 'success',
            'data': processed_markets,
            'total': len(processed_markets),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in markets: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@exchanges_router.get("/fiats", summary="ارزهای فیات")
async def get_fiats():
    """دریافت ارزهای فیات پردازش شده"""
    try:
        raw_data = coin_stats_manager.get_fiats()
        
        processed_fiats = []
        for fiat in raw_data.get('result', []):
            processed_fiats.append({
                'symbol': fiat.get('symbol'),
                'name': fiat.get('name'),
                'symbol_native': fiat.get('symbol_native'),
                'decimal_digits': fiat.get('decimal_digits'),
                'rounding': fiat.get('rounding'),
                'code': fiat.get('code'),
                'name_plural': fiat.get('name_plural')
            })
        
        return {
            'status': 'success',
            'data': processed_fiats,
            'total': len(processed_fiats),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in fiats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@exchanges_router.get("/currencies", summary="ارزها")
async def get_currencies():
    """دریافت ارزهای پردازش شده"""
    try:
        raw_data = coin_stats_manager.get_currencies()
        
        return {
            'status': 'success',
            'data': raw_data.get('result', []),
            'total': len(raw_data.get('result', [])),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in currencies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@exchanges_router.get("/price", summary="قیمت صرافی")
async def get_exchange_price(
    exchange: str = Query("Binance"),
    from_coin: str = Query("BTC"),
    to_coin: str = Query("ETH"),
    timestamp: str = Query("1636315200")
):
    """دریافت قیمت پردازش شده صرافی"""
    try:
        raw_data = coin_stats_manager.get_exchange_price(exchange, from_coin, to_coin, timestamp)
        
        return {
            'status': 'success',
            'data': {
                'exchange': exchange,
                'from_coin': from_coin,
                'to_coin': to_coin,
                'timestamp': timestamp,
                'price': raw_data.get('price'),
                'last_updated': datetime.now().isoformat()
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in exchange price: {e}")
        raise HTTPException(status_code=500, detail=str(e))
