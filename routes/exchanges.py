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
        # دریافت داده خام - ممکن است لیست مستقیم باشد
        raw_data = coin_stats_manager.get_exchanges()
        
        # اگر لیست است، مستقیماً پردازش کنیم
        if isinstance(raw_data, list):
            exchanges_data = raw_data
        else:
            # اگر دیکشنری است، از کلید data استفاده کنیم
            if "error" in raw_data:
                raise HTTPException(status_code=500, detail=raw_data["error"])
            exchanges_data = raw_data.get('data', [])
        
        processed_exchanges = []
        for exchange in exchanges_data:
            processed_exchanges.append({
                'id': exchange.get('id'),
                'name': exchange.get('name'),
                'rank': exchange.get('rank'),
                'percentTotalVolume': exchange.get('percentTotalVolume'),
                'volumeUsd': exchange.get('volumeUsd'),
                'tradingPairs': exchange.get('tradingPairs'),
                'socket': exchange.get('socket'),
                'exchangeUrl': exchange.get('exchangeUrl'),
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
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # استفاده از کلیدهای صحیح از داده خام
        markets_data = raw_data.get('data', [])
        processed_markets = []
        for market in markets_data:
            processed_markets.append({
                'exchange': market.get('exchange'),
                'base_asset': market.get('from'),
                'quote_asset': market.get('to'),
                'pair': market.get('pair'),
                'price': market.get('price'),
                'volume_24h': market.get('volume'),
                'pair_volume': market.get('pairVolume'),
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
        # دریافت داده خام - ممکن است لیست مستقیم باشد
        raw_data = coin_stats_manager.get_fiats()
        
        # اگر لیست است، مستقیماً پردازش کنیم
        if isinstance(raw_data, list):
            fiats_data = raw_data
        else:
            # اگر دیکشنری است، از کلید data استفاده کنیم
            if "error" in raw_data:
                raise HTTPException(status_code=500, detail=raw_data["error"])
            fiats_data = raw_data.get('data', [])
        
        processed_fiats = []
        for fiat in fiats_data:
            processed_fiats.append({
                'symbol': fiat.get('symbol'),
                'name': fiat.get('name'),
                'symbol_native': fiat.get('symbol_native'),
                'decimal_digits': fiat.get('decimal_digits'),
                'rounding': fiat.get('rounding'),
                'code': fiat.get('code'),
                'name_plural': fiat.get('name_plural'),
                'last_updated': datetime.now().isoformat()
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
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # اگر داده لیست است، مستقیماً استفاده کنیم
        if isinstance(raw_data, list):
            currencies_data = raw_data
        else:
            currencies_data = raw_data.get('data', raw_data.get('result', []))
        
        return {
            'status': 'success',
            'data': currencies_data,
            'total': len(currencies_data),
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in currencies: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@exchanges_router.get("/price", summary="قیمت صرافی")
async def get_exchange_price(
    exchange: str = Query("Binance"),
    from_coin: str = Query("BTC"),
    to_coin: str = Query("USDT"),
    timestamp: str = Query(None)
):
    """دریافت قیمت پردازش شده صرافی"""
    try:
        if not timestamp:
            timestamp = str(int(datetime.now().timestamp()))
            
        raw_data = coin_stats_manager.get_exchange_price(exchange, from_coin, to_coin, timestamp)
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # بررسی ساختارهای مختلف برای یافتن قیمت
        price = None
        
        # ساختار 1: قیمت در data.data.price
        if isinstance(raw_data.get('data'), dict):
            price = raw_data['data'].get('price')
        
        # ساختار 2: قیمت در data.price  
        elif 'price' in raw_data:
            price = raw_data.get('price')
            
        # ساختار 3: قیمت مستقیماً در ریشه
        elif 'data' in raw_data and isinstance(raw_data['data'], dict):
            price = raw_data['data'].get('price')
        
        return {
            'status': 'success',
            'data': {
                'exchange': exchange,
                'from_coin': from_coin,
                'to_coin': to_coin,
                'timestamp': timestamp,
                'price': price,
                'last_updated': datetime.now().isoformat()
            },
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error in exchange price: {e}")
        raise HTTPException(status_code=500, detail=str(e))
