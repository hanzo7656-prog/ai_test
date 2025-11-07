from fastapi import APIRouter, HTTPException
from datetime import datetime
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

docs_router = APIRouter(prefix="/api/docs", tags=["Documentation"])

@docs_router.get("/complete", summary="مستندات کامل API")
async def get_complete_docs():
    """مستندات کامل و دقیق تمام اندپوینت‌های VortexAI"""
    
    return {
        "title": "VortexAI API - Complete Documentation",
        "version": "4.0.0",
        "last_updated": datetime.now().isoformat(),
        "description": "مستندات کامل و دقیق سیستم VortexAI با مثال‌های کاربردی",
        
        "📖 Introduction": {
            "description": "VortexAI یک سیستم کامل تحلیل بازار کریپتو با هوش مصنوعی پیشرفته",
            "base_url": "https://your-domain.com",
            "authentication": "Currently no authentication required",
            "rate_limits": "1000 requests per hour per IP",
            "response_format": "All responses are in JSON format"
        },
        
        "🚀 Quick Start": {
            "description": "شروع سریع با اندپوینت‌های اصلی",
            "examples": {
                "check_health": {
                    "method": "GET",
                    "url": "/api/health/status",
                    "description": "بررسی سلامت سیستم"
                },
                "get_bitcoin": {
                    "method": "GET", 
                    "url": "/api/coins/details/bitcoin",
                    "description": "دریافت اطلاعات بیت‌کوین"
                },
                "get_news": {
                    "method": "GET",
                    "url": "/api/news/all?limit=5",
                    "description": "دریافت آخرین اخبار"
                }
            }
        },
        
        "💰 Coins API - پردازش شده": {
            "description": "داده‌های نمادهای ارز دیجیتال با پردازش و تحلیل",
            
            "get_coin_list": {
                "method": "GET",
                "url": "/api/coins/list",
                "description": "دریافت لیست نمادها با قابلیت صفحه‌بندی و مرتب‌سازی",
                "parameters": {
                    "limit": "تعداد نتایج (پیش‌فرض: 20, حداکثر: 100)",
                    "page": "شماره صفحه (پیش‌فرض: 1)",
                    "currency": "ارز پایه (پیش‌فرض: USD)",
                    "sort_by": "فیلد مرتب‌سازی (پیش‌فرض: rank)"
                },
                "example_request": "GET /api/coins/list?limit=10&page=1&currency=USD&sort_by=price",
                "example_response": {
                    "status": "success",
                    "data": [
                        {
                            "id": "bitcoin",
                            "name": "Bitcoin",
                            "symbol": "BTC",
                            "price": 45000.50,
                            "price_change_24h": 2.5,
                            "volume_24h": 28500000000,
                            "market_cap": 880000000000,
                            "rank": 1,
                            "analysis": {
                                "trend": "uptrend",
                                "signal": "BUY",
                                "confidence": 0.75
                            }
                        }
                    ],
                    "pagination": {
                        "page": 1,
                        "limit": 10,
                        "total": 100
                    }
                }
            },
            
            "get_coin_details": {
                "method": "GET",
                "url": "/api/coins/details/{coin_id}",
                "description": "دریافت جزئیات کامل یک نماد خاص",
                "parameters": {
                    "coin_id": "شناسه نماد (مثال: bitcoin, ethereum)",
                    "currency": "ارز پایه (پیش‌فرض: USD)"
                },
                "example_request": "GET /api/coins/details/bitcoin?currency=USD",
                "example_response": {
                    "status": "success",
                    "data": {
                        "id": "bitcoin",
                        "name": "Bitcoin",
                        "symbol": "BTC",
                        "price": 45000.50,
                        "price_change_24h": 2.5,
                        "price_change_1h": 0.3,
                        "price_change_1w": 5.2,
                        "volume_24h": 28500000000,
                        "market_cap": 880000000000,
                        "rank": 1,
                        "website": "https://bitcoin.org",
                        "description": "Bitcoin is a decentralized digital currency...",
                        "analysis": {
                            "trend": "uptrend",
                            "signal": "BUY", 
                            "confidence": 0.75
                        }
                    }
                }
            },
            
            "get_coin_charts": {
                "method": "GET",
                "url": "/api/coins/charts/{coin_id}",
                "description": "دریافت داده‌های چارت برای تحلیل تکنیکال",
                "parameters": {
                    "coin_id": "شناسه نماد",
                    "period": "بازه زمانی (24h, 1w, 1m, 3m, 6m, 1y, all - پیش‌فرض: 1w)"
                },
                "example_request": "GET /api/coins/charts/bitcoin?period=1w",
                "example_response": {
                    "status": "success",
                    "data": {
                        "coin_id": "bitcoin",
                        "period": "1w",
                        "prices": [
                            [1638316800000, 45000.50],
                            [1638403200000, 45500.75],
                            # ...
                        ],
                        "analysis": {
                            "trend": "uptrend",
                            "volatility": 2.5,
                            "support_resistance": {
                                "support": 44500.00,
                                "resistance": 46000.00
                            }
                        }
                    }
                }
            }
        },
        
        "📊 Raw Coins API - داده‌های خام": {
            "description": "داده‌های خام نمادها بدون پردازش - مناسب برای هوش مصنوعی و تحلیل‌های پیشرفته",
            
            "get_raw_coin_details": {
                "method": "GET", 
                "url": "/api/raw/coins/details/{coin_id}",
                "description": "دریافت داده‌های خام یک نماد - دقیقاً مطابق CoinStats API",
                "example_request": "GET /api/raw/coins/details/bitcoin",
                "example_response": {
                    "status": "success",
                    "data_type": "raw",
                    "source": "coinstats_api",
                    "coin_id": "bitcoin",
                    "data": {
                        "id": "bitcoin",
                        "name": "Bitcoin", 
                        "symbol": "BTC",
                        "price": 45000.50,
                        "priceChange1d": 2.5,
                        "priceChange1h": 0.3,
                        "priceChange1w": 5.2,
                        "volume": 28500000000,
                        "marketCap": 880000000000,
                        "rank": 1,
                        "websiteUrl": "https://bitcoin.org",
                        "description": "Bitcoin is a decentralized digital currency...",
                        "links": [
                            {
                                "name": "website",
                                "url": "https://bitcoin.org",
                                "type": "website"
                            }
                        ]
                        # ... تمام فیلدهای اصلی CoinStats API
                    }
                }
            }
        },
        
        "📰 News API - اخبار و تحلیل": {
            "description": "اخبار و تحلیل‌های بازار کریپتو",
            
            "get_news": {
                "method": "GET",
                "url": "/api/news/all",
                "description": "دریافت آخرین اخبار بازار",
                "parameters": {
                    "limit": "تعداد اخبار (پیش‌فرض: 50, حداکثر: 100)"
                },
                "example_response": {
                    "status": "success", 
                    "data": [
                        {
                            "id": "news_123",
                            "title": "Bitcoin Reaches New All-Time High",
                            "description": "Bitcoin price surges to $45,000...",
                            "url": "https://example.com/news/123",
                            "source": "CoinTelegraph",
                            "published_at": "2024-01-15T10:30:00Z",
                            "sentiment": "bullish",
                            "importance": 4,
                            "tags": ["bitcoin", "price", "bullish"]
                        }
                    ]
                }
            }
        },
        
        "🔍 Insights API - بینش و تحلیل بازار": {
            "description": "تحلیل‌های پیشرفته بازار و شاخص‌ها",
            
            "get_fear_greed": {
                "method": "GET",
                "url": "/api/insights/fear-greed", 
                "description": "شاخص ترس و طمع بازار کریپتو",
                "example_response": {
                    "status": "success",
                    "data": {
                        "value": 65,
                        "value_classification": "Greed",
                        "timestamp": "2024-01-15T10:30:00Z",
                        "analysis": {
                            "current_sentiment": "Greed",
                            "market_condition": "Greed - Bullish sentiment",
                            "risk_level": "Medium",
                            "suggested_action": "Monitor for entry points"
                        }
                    }
                }
            }
        },
        
        "⚡ Health & Debug API": {
            "description": "مانیتورینگ و دیباگ سیستم",
            
            "get_health_status": {
                "method": "GET",
                "url": "/api/health/status",
                "description": "بررسی سلامت کامل سیستم",
                "example_response": {
                    "system": "operational",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "subsystems": {
                        "api_endpoints": "healthy",
                        "debug_system": "active", 
                        "database": "healthy"
                    },
                    "key_metrics": {
                        "response_time_avg": "45ms",
                        "uptime": "15 days, 2:30:15",
                        "active_connections": 45
                    }
                }
            }
        },
        
        "🛠️ Common Parameters": {
            "currency": {
                "description": "ارز پایه برای قیمت‌ها",
                "default": "USD",
                "supported": ["USD", "EUR", "GBP", "JPY", "CAD", "AUD"]
            },
            "pagination": {
                "description": "پارامترهای صفحه‌بندی",
                "limit": "تعداد آیتم در هر صفحه (1-100)",
                "page": "شماره صفحه (از 1 شروع می‌شود)"
            }
        },
        
        "❌ Error Handling": {
            "description": "مدیریت خطاهای سیستم",
            "common_errors": {
                "400": "درخواست نامعتبر - پارامترهای ورودی را بررسی کنید",
                "404": "منبع یافت نشد - آدرس را بررسی کنید", 
                "429": "تعداد درخواست بیش از حد - لطفاً کمی صبر کنید",
                "500": "خطای سرور - با پشتیبانی تماس بگیرید",
                "503": "سرویس در دسترس نیست - سرویس خارجی قطع شده"
            },
            "error_response_format": {
                "error": "ERROR_CODE",
                "message": "شرح خطا به زبان انسانی",
                "status_code": 400,
                "timestamp": "2024-01-15T10:30:00Z",
                "details": "اطلاعات اضافی برای دیباگ"
            }
        },
        
        "🔗 Useful Links": {
            "interactive_docs": "/docs",
            "roadmap": "/api/roadmap", 
            "quick_reference": "/api/quick-reference",
            "health_check": "/api/health/status",
            "github_repository": "https://github.com/your-repo/vortexai"
        }
    }

@docs_router.get("/coins", summary="مستندات کامل Coin API")
async def get_coins_docs():
    """مستندات تخصصی بخش نمادها"""
    return {
        "section": "Coins API Documentation",
        "description": "مستندات کامل و تخصصی API نمادهای ارز دیجیتال",
        "last_updated": datetime.now().isoformat(),
        
        "endpoints": {
            "list_coins": {
                "url": "/api/coins/list",
                "method": "GET",
                "description": "دریافت لیست نمادها با قابلیت فیلتر و مرتب‌سازی",
                "parameters": {
                    "limit": {"type": "integer", "default": 20, "min": 1, "max": 100},
                    "page": {"type": "integer", "default": 1, "min": 1},
                    "currency": {"type": "string", "default": "USD", "options": ["USD", "EUR", "GBP"]},
                    "sort_by": {"type": "string", "default": "rank", "options": ["rank", "price", "volume", "marketCap"]},
                    "sort_dir": {"type": "string", "default": "asc", "options": ["asc", "desc"]}
                },
                "response_fields": {
                    "status": "وضعیت درخواست (success/error)",
                    "data": "آرایه‌ای از نمادها",
                    "pagination": "اطلاعات صفحه‌بندی"
                }
            }
            # ... سایر اندپوینت‌ها به همین صورت
        }
    }

@docs_router.get("/examples", summary="مثال‌های کاربردی")
async def get_code_examples():
    """مثال‌های کد برای استفاده از API"""
    return {
        "title": "Code Examples - VortexAI API",
        "last_updated": datetime.now().isoformat(),
        
        "javascript_fetch": {
            "description": "استفاده با Fetch API در JavaScript",
            "code": """
// دریافت اطلاعات بیت‌کوین
async function getBitcoinData() {
    try {
        const response = await fetch('/api/coins/details/bitcoin');
        const data = await response.json();
        
        if (data.status === 'success') {
            console.log('Bitcoin Price:', data.data.price);
            console.log('24h Change:', data.data.price_change_24h);
            console.log('Signal:', data.data.analysis.signal);
        }
    } catch (error) {
        console.error('Error fetching data:', error);
    }
}

// دریافت لیست نمادها
async function getCoinsList(limit = 10) {
    const response = await fetch(\`/api/coins/list?limit=\${limit}\`);
    return await response.json();
}
            """
        },
        
        "python_requests": {
            "description": "استفاده با Requests در Python",
            "code": """
import requests

def get_coin_details(coin_id):
    url = f"https://your-domain.com/api/coins/details/{coin_id}"
    response = requests.get(url)
    
    if response.status_code == 200:
        data = response.json()
        if data['status'] == 'success':
            return data['data']
    else:
        print(f"Error: {response.status_code}")
        return None

# مثال استفاده
bitcoin_data = get_coin_details('bitcoin')
if bitcoin_data:
    print(f"Bitcoin Price: {bitcoin_data['price']}")
    print(f"Signal: {bitcoin_data['analysis']['signal']}")
            """
        },
        
        "curl_examples": {
            "description": "دستورات cURL برای تست سریع",
            "code": """
# بررسی سلامت سیستم
curl -X GET "https://your-domain.com/api/health/status"

# دریافت اطلاعات بیت‌کوین
curl -X GET "https://your-domain.com/api/coins/details/bitcoin"

# دریافت لیست ۱۰ نماد برتر
curl -X GET "https://your-domain.com/api/coins/list?limit=10"

# دریافت اخبار
curl -X GET "https://your-domain.com/api/news/all?limit=5"
            """
        }
    }
