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
        processed_data = coin_stats_manager.get_exchanges_processed()
        
        if "error" in processed_data:
            raise HTTPException(status_code=500, detail=processed_data["error"])
        
        return processed_data
        
    except Exception as e:
        logger.error(f"Error in exchanges list: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@exchanges_router.get("/markets", summary="مارکت‌ها")
async def get_markets():
    """دریافت مارکت‌های پردازش شده"""
    try:
        raw_data = coin_stats_manager.get_markets()
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        processed_markets = []
        for market in raw_data.get('data', []):
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
            'raw_data': raw_data,
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
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        processed_fiats = []
        for fiat in raw_data.get('data', []):
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
            'raw_data': raw_data,
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
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        return {
            'status': 'success',
            'data': raw_data.get('result', []),
            'total': len(raw_data.get('result', [])),
            'raw_data': raw_data,
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
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
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
            'raw_data': raw_data,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in exchange price: {e}")
        raise HTTPException(status_code=500, detail=str(e))
