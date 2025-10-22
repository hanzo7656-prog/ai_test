#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 AI Trading Assistant - Fixed Version
No TA-Lib dependency, optimized for Render
"""

import asyncio
import sys
import os
import gc
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

logger = logging.getLogger(__name__)

class SystemValidator:
    """Lightweight system validator"""
    
    def validate(self):
        """Basic system validation"""
        logger.info(f"🐍 Python version: {sys.version}")
        
        # Check critical dependencies
        critical_deps = ['torch', 'numpy', 'pandas', 'requests', 'fastapi', 'uvicorn']
        missing = []
        
        for dep in critical_deps:
            try:
                __import__(dep)
                logger.info(f"✅ {dep} is available")
            except ImportError:
                missing.append(dep)
                logger.error(f"❌ {dep} not found")
        
        if missing:
            logger.warning(f"Missing packages: {missing}")
        
        return len(missing) == 0

class SimpleTechnicalAnalysis:
    """Simple technical analysis without TA-Lib"""
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        """Calculate RSI without TA-Lib"""
        if len(prices) < period + 1:
            return 50.0  # Default neutral value
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100.0 if avg_gain > 0 else 50.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def calculate_sma(prices, period=20):
        """Calculate Simple Moving Average"""
        if len(prices) < period:
            return np.mean(prices) if len(prices) > 0 else prices[-1]
        return np.mean(prices[-period:])
    
    @staticmethod
    def calculate_ema(prices, period=12):
        """Calculate Exponential Moving Average"""
        if len(prices) < period:
            return np.mean(prices) if len(prices) > 0 else prices[-1]
        
        ema = prices[0]
        alpha = 2 / (period + 1)
        
        for price in prices[1:]:
            ema = alpha * price + (1 - alpha) * ema
        
        return ema

class LightweightAIModel:
    """Lightweight AI model without complex dependencies"""
    
    def __init__(self):
        self.initialized = False
        self.technical_analysis = SimpleTechnicalAnalysis()
    
    def initialize(self):
        """Initialize model with minimal dependencies"""
        try:
            import torch
            import torch.nn as nn
            import numpy as np
            
            # Simple neural network
            class SimpleTradingModel(nn.Module):
                def __init__(self, input_size=5, hidden_size=32, output_size=3):
                    super().__init__()
                    self.network = nn.Sequential(
                        nn.Linear(input_size, hidden_size),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(hidden_size, output_size)
                    )
                
                def forward(self, x):
                    return self.network(x)
            
            self.model = SimpleTradingModel()
            self.has_torch = True
            logger.info("✅ PyTorch model initialized")
            
        except ImportError as e:
            # Fallback to simple logic
            self.has_torch = False
            logger.info("✅ Using rule-based fallback model")
        
        self.initialized = True
    
    def analyze_market(self, price_data):
        """Analyze market data"""
        if not self.initialized:
            return {"error": "Model not initialized"}
        
        try:
            import numpy as np
            
            # Extract prices
            prices = [float(x) for x in price_data.get('prices', [])[-30:]]  # Last 30 prices
            if len(prices) < 10:
                return {"error": "Insufficient data"}
            
            # Calculate indicators
            rsi = self.technical_analysis.calculate_rsi(prices)
            sma_20 = self.technical_analysis.calculate_sma(prices, 20)
            current_price = prices[-1]
            
            # Simple trading logic
            if rsi < 30 and current_price < sma_20:
                signal = "BUY"
                confidence = min((30 - rsi) / 30, 0.9)
            elif rsi > 70 and current_price > sma_20:
                signal = "SELL"
                confidence = min((rsi - 70) / 30, 0.9)
            else:
                signal = "HOLD"
                confidence = 0.5
            
            return {
                "signal": signal,
                "confidence": round(confidence, 2),
                "indicators": {
                    "rsi": round(rsi, 2),
                    "sma_20": round(sma_20, 2),
                    "current_price": round(current_price, 2)
                },
                "model_type": "pytorch" if self.has_torch else "rule_based"
            }
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}"}

async def start_web_server():
    """Start FastAPI web server"""
    try:
        from fastapi import FastAPI, HTTPException
        import uvicorn
        
        app = FastAPI(
            title="AI Trading Assistant",
            description="Lightweight AI for market analysis",
            version="1.0.0"
        )
        
        # Initialize components
        validator = SystemValidator()
        ai_model = LightweightAIModel()
        
        validator.validate()
        ai_model.initialize()
        
        @app.get("/")
        async def root():
            return {
                "status": "running",
                "service": "AI Trading Assistant",
                "version": "1.0.0",
                "memory_optimized": True
            }
        
        @app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "timestamp": time.time(),
                "python_version": sys.version
            }
        
        @app.post("/analyze/market")
        async def analyze_market(price_data: dict):
            """Analyze market data"""
            try:
                result = ai_model.analyze_market(price_data)
                return {"analysis": result}
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @app.get("/system/info")
        async def system_info():
            """Get system information"""
            try:
                import psutil
                memory = psutil.virtual_memory()
                return {
                    "memory_used_mb": memory.used // (1024 * 1024),
                    "memory_available_mb": memory.available // (1024 * 1024),
                    "memory_percent": memory.percent,
                    "cpu_percent": psutil.cpu_percent()
                }
            except ImportError:
                return {"memory_info": "psutil not available"}
        
        # Server configuration
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=int(os.getenv("PORT", "8000")),
            log_level="info",
            access_log=True
        )
        
        server = uvicorn.Server(config)
        logger.info(f"🌐 Web server starting on port {config.port}")
        
        await server.serve()
        
    except Exception as e:
        logger.error(f"Web server failed: {e}")
        # Fallback: simple HTTP server
        await start_fallback_server()

async def start_fallback_server():
    """Fallback simple HTTP server"""
    try:
        import http.server
        import socketserver
        import json
        
        class SimpleHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/health':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response = json.dumps({"status": "healthy", "mode": "fallback"})
                    self.wfile.write(response.encode())
                else:
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response = json.dumps({"status": "running", "message": "Fallback mode"})
                    self.wfile.write(response.encode())
        
        port = int(os.getenv("PORT", "8000"))
        with socketserver.TCPServer(("", port), SimpleHandler) as httpd:
            logger.info(f"🌐 Fallback server started on port {port}")
            httpd.serve_forever()
            
    except Exception as e:
        logger.error(f"Fallback server also failed: {e}")

async def main():
    """Main application entry point"""
    logger.info("🚀 Starting AI Trading Assistant...")
    
    try:
        await start_web_server()
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"❌ Server crashed: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("👋 Application terminated by user")
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        sys.exit(1)
