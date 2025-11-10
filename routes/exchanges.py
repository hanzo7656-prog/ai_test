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
        # دریافت مستقیم از API
        raw_data = coin_stats_manager._make_api_request("tickers/exchanges")
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # پردازش داده‌ها - استفاده از ساختار واقعی API
        exchanges_data = raw_data.get("data", raw_data.get("result", []))
        
        # اگر داده لیست مستقیم است
        if isinstance(exchanges_data, list):
            processed_exchanges = []
            for exchange in exchanges_data:
                # بررسی ساختارهای مختلف داده
                if isinstance(exchange, dict):
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
                else:
                    # اگر داده ساده است
                    processed_exchanges.append({
                        'name': str(exchange),
                        'last_updated': datetime.now().isoformat()
                    })
            
            return {
                'status': 'success',
                'data': processed_exchanges,
                'total': len(processed_exchanges),
                'timestamp': datetime.now().isoformat()
            }
        else:
            # اگر داده لیست نیست
            return {
                'status': 'success',
                'data': [],
                'total': 0,
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
        
        # اگر داده لیست مستقیم است
        if isinstance(raw_data, list):
            markets_data = raw_data
        else:
            if "error" in raw_data:
                raise HTTPException(status_code=500, detail=raw_data["error"])
            markets_data = raw_data.get('data', raw_data.get('result', []))
        
        processed_markets = []
        for market in markets_data:
            if isinstance(market, dict):
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
        # دریافت مستقیم از API
        raw_data = coin_stats_manager._make_api_request("fiats")
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # پردازش داده‌ها - استفاده از ساختار واقعی API
        fiats_data = raw_data.get("data", raw_data.get("result", []))
        
        # اگر داده لیست مستقیم است
        if isinstance(fiats_data, list):
            processed_fiats = []
            for fiat in fiats_data:
                if isinstance(fiat, dict):
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
        else:
            # اگر داده لیست نیست
            return {
                'status': 'success',
                'data': [],
                'total': 0,
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
        
        # اگر داده لیست مستقیم است
        if isinstance(raw_data, list):
            currencies_data = raw_data
        else:
            if "error" in raw_data:
                raise HTTPException(status_code=500, detail=raw_data["error"])
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
            
        # دریافت مستقیم از API
        params = {
            "exchange": exchange,
            "from": from_coin,
            "to": to_coin,
            "timestamp": timestamp
        }
        raw_data = coin_stats_manager._make_api_request("coins/price/exchange", params)
        
        if "error" in raw_data:
            raise HTTPException(status_code=500, detail=raw_data["error"])
        
        # پردازش قیمت - استفاده از ساختار واقعی API
        price_data = raw_data.get("data", {})
        price = price_data.get('price')
        
        # اگر قیمت پیدا نشد، از ساختارهای دیگر جستجو کن
        if price is None:
            price = raw_data.get('price')
        
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
