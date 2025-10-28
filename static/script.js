// API Base URL
const API_BASE = 'https://ai-test-cdwg.onrender.com';

// DOM Elements
const elements = {
    btcPrice: document.getElementById('btc-price'),
    ethPrice: document.getElementById('eth-price'),
    fearGreed: document.getElementById('fear-greed'),
    symbolsInput: document.getElementById('symbols-input'),
    periodSelect: document.getElementById('period-select'),
    analysisResults: document.getElementById('analysis-results'),
    realtimeData: document.getElementById('realtime-data'),
    cpuUsage: document.getElementById('cpu-usage'),
    memoryUsage: document.getElementById('memory-usage'),
    apiCalls: document.getElementById('api-calls')
};

// Initialize Dashboard
async function initDashboard() {
    await loadMarketData();
    await loadSystemStatus();
    startWebSocketUpdates();
}

// Load Market Data
async function loadMarketData() {
    try {
        const response = await fetch(`${API_BASE}/market/overview`);
        const data = await response.json();
        
        if (data.success) {
            updateMarketDisplay(data);
        }
    } catch (error) {
        console.error('Error loading market data:', error);
    }
}

// Update Market Display
function updateMarketDisplay(data) {
    if (data.current_price) {
        elements.btcPrice.textContent = `$${data.current_price.toLocaleString()}`;
    }
    
    // Update other market data points...
}

// Run AI Analysis
async function runAIAnalysis() {
    const symbols = elements.symbolsInput.value;
    const period = elements.periodSelect.value;
    
    // Show loading
    elements.analysisResults.innerHTML = `
        <div class="result-placeholder">
            <div class="loading"></div>
            <p>AI is analyzing market data...</p>
        </div>
    `;
    
    try {
        const response = await fetch(`${API_BASE}/ai/analysis?symbols=${symbols}&period=${period}`);
        const data = await response.json();
        
        displayAnalysisResults(data);
    } catch (error) {
        console.error('Error running AI analysis:', error);
        elements.analysisResults.innerHTML = `
            <div class="result-placeholder">
                <i class="ri-error-warning-line"></i>
                <p>Error running analysis</p>
            </div>
        `;
    }
}

// Display Analysis Results
function displayAnalysisResults(data) {
    if (data.status === 'success' && data.analysis_report) {
        const report = data.analysis_report;
        
        let html = `
            <div class="analysis-summary">
                <h3>AI Analysis Results</h3>
                <div class="symbols-grid">
        `;
        
        Object.entries(report.symbol_analysis).forEach(([symbol, analysis]) => {
            const signal = analysis.ai_signal?.signals;
            html += `
                <div class="symbol-card">
                    <h4>${symbol}</h4>
                    <div class="signal ${signal?.primary_signal?.toLowerCase()}">
                        ${signal?.primary_signal || 'HOLD'}
                    </div>
                    <div class="confidence">
                        Confidence: ${((signal?.signal_confidence || 0) * 100).toFixed(1)}%
                    </div>
                    <div class="price">$${analysis.current_price || 0}</div>
                </div>
            `;
        });
        
        html += `
                </div>
            </div>
        `;
        
        elements.analysisResults.innerHTML = html;
    }
}

// Load System Status
async function loadSystemStatus() {
    try {
        const response = await fetch(`${API_BASE}/system/health`);
        const data = await response.json();
        
        if (data.system_health) {
            elements.cpuUsage.textContent = `${Math.round(data.system_health.cpu.percent)}%`;
            elements.memoryUsage.textContent = `${Math.round(data.system_health.memory.percent)}%`;
        }
        
        if (data.api_statistics) {
            elements.apiCalls.textContent = data.api_statistics.total_api_calls;
        }
    } catch (error) {
        console.error('Error loading system status:', error);
    }
}

// WebSocket Updates (Simulated)
function startWebSocketUpdates() {
    // Simulate real-time updates
    setInterval(() => {
        updateRealtimeData();
    }, 5000);
}

function updateRealtimeData() {
    // This would connect to your WebSocket endpoint
    // For now, we'll simulate some data
    const pairs = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT'];
    
    let html = '';
    pairs.forEach(pair => {
        const price = (Math.random() * 100000).toFixed(2);
        const change = (Math.random() * 10 - 5).toFixed(2);
        const isPositive = parseFloat(change) > 0;
        
        html += `
            <div class="pair-item">
                <div class="pair-name">${pair}</div>
                <div class="pair-price">$${price}</div>
                <div class="pair-change ${isPositive ? 'positive' : 'negative'}">
                    ${isPositive ? '+' : ''}${change}%
                </div>
            </div>
        `;
    });
    
    elements.realtimeData.innerHTML = html;
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', initDashboard);
