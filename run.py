#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 AI Trading Assistant - Main Entry Point
Optimized for Python 3.13 and 512MB RAM environment
"""

import asyncio
import sys
import os
import gc
import logging
from pathlib import Path
from typing import Optional, Dict, Any

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configure logging before importing other modules
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('ai_trading_assistant.log', encoding='utf-8')
    ]
)

logger = logging.getLogger(__name__)

class SystemValidator:
    """Validate system requirements and dependencies"""
    
    def __init__(self):
        self.requirements_met = False
        self.system_info = {}
    
    def validate_python_version(self):
        """Validate Python version"""
        version = sys.version_info
        if version.major == 3 and version.minor >= 13:
            logger.info(f"✅ Python version: {sys.version}")
            return True
        else:
            logger.error(f"❌ Python 3.13+ required. Current: {sys.version}")
            return False
    
    def validate_memory(self):
        """Validate available memory"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            self.system_info['total_memory_gb'] = memory.total / (1024**3)
            self.system_info['available_memory_gb'] = memory.available / (1024**3)
            
            logger.info(f"💾 Total RAM: {self.system_info['total_memory_gb']:.1f}GB")
            logger.info(f"📊 Available RAM: {self.system_info['available_memory_gb']:.1f}GB")
            
            if self.system_info['available_memory_gb'] < 0.4:  # 400MB
                logger.warning("⚠️  Low memory available. Performance may be affected.")
            
            return True
        except ImportError:
            logger.warning("⚠️  psutil not available, memory validation skipped")
            return True
    
    def validate_dependencies(self):
        """Validate critical dependencies"""
        critical_deps = ['torch', 'numpy', 'pandas', 'requests']
        missing_deps = []
        
        for dep in critical_deps:
            try:
                __import__(dep)
                logger.info(f"✅ {dep} is available")
            except ImportError as e:
                missing_deps.append(dep)
                logger.error(f"❌ {dep} not available: {e}")
        
        if missing_deps:
            logger.error(f"Missing critical dependencies: {missing_deps}")
            return False
        return True
    
    def run_validation(self):
        """Run all validations"""
        logger.info("🔍 Starting system validation...")
        
        checks = [
            self.validate_python_version(),
            self.validate_memory(),
            self.validate_dependencies()
        ]
        
        self.requirements_met = all(checks)
        
        if self.requirements_met:
            logger.info("✅ All system requirements met!")
        else:
            logger.error("❌ System validation failed!")
        
        return self.requirements_met

class AIApplication:
    """Main AI Trading Application"""
    
    def __init__(self):
        self.validator = SystemValidator()
        self.is_running = False
        self.modules = {}
        
    async def initialize_modules(self):
        """Initialize all AI modules with memory optimization"""
        logger.info("🔄 Initializing AI modules...")
        
        try:
            # Import modules dynamically to manage memory
            from memory_efficient_spike_transformer import RenderFreeSpikeTransformer, MemoryMonitor
            from api_integration import DataManager
            from technical_engine import AdvancedTechnicalEngine
            
            # Initialize with optimized settings
            self.modules['memory_monitor'] = MemoryMonitor()
            self.modules['memory_monitor'].print_usage("Before module initialization:")
            
            # Initialize AI model with minimal memory footprint
            self.modules['ai_model'] = RenderFreeSpikeTransformer(
                vocab_size=3000,
                d_model=96,
                n_heads=3,
                num_layers=2,
                d_ff=192,
                max_seq_len=96,
                num_classes=3
            )
            
            # Initialize data manager
            self.modules['data_manager'] = DataManager(raw_data_path="./raw_data")
            
            # Initialize technical engine
            self.modules['technical_engine'] = AdvancedTechnicalEngine()
            
            # Force garbage collection
            gc.collect()
            self.modules['memory_monitor'].print_usage("After module initialization:")
            
            logger.info("✅ AI modules initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize modules: {e}")
            return False
    
    async def start_web_server(self):
        """Start FastAPI web server"""
        try:
            from fastapi import FastAPI, HTTPException
            from fastapi.middleware.cors import CORSMiddleware
            import uvicorn
            
            app = FastAPI(
                title="AI Trading Assistant",
                description="Ultra-efficient AI for crypto analysis",
                version="1.0.0"
            )
            
            # CORS middleware
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
            @app.get("/")
            async def root():
                return {
                    "status": "running",
                    "message": "AI Trading Assistant API",
                    "version": "1.0.0"
                }
            
            @app.get("/health")
            async def health_check():
                """Health check endpoint"""
                return {
                    "status": "healthy",
                    "memory_usage": self.modules['memory_monitor'].get_memory_usage(),
                    "modules_loaded": list(self.modules.keys())
                }
            
            @app.post("/analyze/text")
            async def analyze_text(text: str):
                """Analyze text sentiment"""
                try:
                    # Simple text analysis implementation
                    result = await self.analyze_text_sentiment(text)
                    return {"analysis": result}
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
            
            @app.get("/system/status")
            async def system_status():
                """Get system status"""
                return {
                    "system_info": self.validator.system_info,
                    "memory_usage": self.modules['memory_monitor'].get_memory_usage(),
                    "model_info": {
                        "parameters": sum(p.numel() for p in self.modules['ai_model'].parameters()),
                        "layers": len(self.modules['ai_model'].layers)
                    }
                }
            
            # Start server in background
            config = uvicorn.Config(
                app,
                host="0.0.0.0",
                port=8000,
                log_level="info",
                access_log=True
            )
            
            server = uvicorn.Server(config)
            
            # Run server in background task
            asyncio.create_task(server.serve())
            
            logger.info("🌐 Web server started on http://0.0.0.0:8000")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to start web server: {e}")
            return False
    
    async def analyze_text_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze text sentiment (placeholder implementation)"""
        # This is a simplified version - implement your actual analysis here
        return {
            "sentiment": "positive",  # Placeholder
            "confidence": 0.85,       # Placeholder
            "tokens_processed": len(text.split()),
            "processing_time_ms": 150
        }
    
    async def data_collection_loop(self):
        """Background data collection loop"""
        logger.info("📊 Starting data collection loop...")
        
        while self.is_running:
            try:
                # Collect and process data
                if 'data_manager' in self.modules:
                    # Example: Collect comprehensive data every 5 minutes
                    consolidated_data = self.modules['data_manager'].collect_comprehensive_data()
                    
                    logger.info(f"📈 Data collected: {len(consolidated_data.get('collected_data', {}))} items")
                
                # Wait before next collection
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"❌ Error in data collection: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retry
    
    async def start_services(self):
        """Start all background services"""
        logger.info("🚀 Starting AI services...")
        
        services = [
            self.start_web_server(),
            self.data_collection_loop()
        ]
        
        # Start all services
        results = await asyncio.gather(*services, return_exceptions=True)
        
        # Check for failures
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"❌ Service {i} failed: {result}")
        
        logger.info("✅ All services started!")
    
    async def run(self):
        """Main application entry point"""
        logger.info("🎯 Starting AI Trading Assistant...")
        
        # Validate system
        if not self.validator.run_validation():
            logger.error("❌ System validation failed. Exiting.")
            return False
        
        # Initialize modules
        if not await self.initialize_modules():
            logger.error("❌ Module initialization failed. Exiting.")
            return False
        
        # Set running flag
        self.is_running = True
        
        try:
            # Start background services
            await self.start_services()
            
            # Main loop
            logger.info("🔄 Entering main application loop...")
            
            while self.is_running:
                try:
                    # Monitor system resources
                    if 'memory_monitor' in self.modules:
                        self.modules['memory_monitor'].print_usage("Main loop:")
                    
                    # Wait before next iteration
                    await asyncio.sleep(60)  # Check every minute
                    
                except KeyboardInterrupt:
                    logger.info("🛑 Keyboard interrupt received. Shutting down...")
                    break
                except Exception as e:
                    logger.error(f"❌ Error in main loop: {e}")
                    await asyncio.sleep(10)
        
        except Exception as e:
            logger.error(f"❌ Critical error in application: {e}")
        
        finally:
            await self.shutdown()
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("🔴 Shutting down AI Trading Assistant...")
        self.is_running = False
        
        # Cleanup resources
        if 'memory_monitor' in self.modules:
            self.modules['memory_monitor'].print_usage("Final shutdown:")
        
        # Force garbage collection
        gc.collect()
        
        logger.info("✅ AI Trading Assistant shutdown complete.")

def main():
    """Main function"""
    try:
        # Create and run application
        app = AIApplication()
        
        # Run async application
        asyncio.run(app.run())
        
    except KeyboardInterrupt:
        logger.info("🛑 Application interrupted by user")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)
    finally:
        logger.info("👋 Application terminated")

if __name__ == "__main__":
    main()
