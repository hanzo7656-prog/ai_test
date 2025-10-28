// مدیریت نمودارها
let charts = {};

// مقداردهی اولیه نمودارها
function initializeCharts() {
    createBTCChart();
    createETHChart();
    // سایر نمودارها...
}

// ایجاد نمودار بیت‌کوین
function createBTCChart() {
    const ctx = document.getElementById('btcChart');
    if (!ctx) return;
    
    charts.btc = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'BTC/USDT',
                data: [],
                borderColor: '#f59e0b',
                backgroundColor: 'rgba(245, 158, 11, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    rtl: true
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        display: false
                    }
                },
                y: {
                    display: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    },
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toLocaleString();
                        }
                    }
                }
            },
            interaction: {
                mode: 'nearest',
                axis: 'x',
                intersect: false
            }
        }
    });
    
    loadChartData('BTC', charts.btc);
}

// ایجاد نمودار اتریوم
function createETHChart() {
    const ctx = document.getElementById('ethChart');
    if (!ctx) return;
    
    charts.eth = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'ETH/USDT',
                data: [],
                borderColor: '#8b5cf6',
                backgroundColor: 'rgba(139, 92, 246, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    mode: 'index',
                    intersect: false,
                    rtl: true
                }
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        display: false
                    }
                },
                y: {
                    display: true,
                    grid: {
                        color: 'rgba(0, 0, 0, 0.1)'
                    },
                    ticks: {
                        callback: function(value) {
                            return '$' + value.toLocaleString();
                        }
                    }
                }
            }
        }
    });
    
    loadChartData('ETH', charts.eth);
}

// بارگذاری داده‌های نمودار
async function loadChartData(symbol, chart) {
    try {
        const data = await apiCall(`/coins/${symbol}/charts?period=7d`);
        
        if (data && data.result) {
            updateChartWithData(chart, data.result, symbol);
        }
    } catch (error) {
        console.error(`Error loading chart data for ${symbol}:`, error);
        // استفاده از داده‌های نمونه
        useSampleChartData(chart, symbol);
    }
}

// به‌روزرسانی نمودار با داده‌های واقعی
function updateChartWithData(chart, chartData, symbol) {
    const prices = [];
    const labels = [];
    
    chartData.forEach((item, index) => {
        if (item.price) {
            prices.push(item.price);
            
            // ایجاد لیبل‌های زمانی
            if (index % Math.ceil(chartData.length / 10) === 0) {
                const date = new Date(item.time || item.timestamp);
                labels.push(date.toLocaleTimeString('fa-IR'));
            } else {
                labels.push('');
            }
        }
    });
    
    chart.data.labels = labels;
    chart.data.datasets[0].data = prices;
    chart.update();
}

// استفاده از داده‌های نمونه
function useSampleChartData(chart, symbol) {
    const sampleData = generateSampleData(50, symbol === 'BTC' ? 45000 : 2500);
    const labels = sampleData.map((_, i) => {
        if (i % 5 === 0) {
            return new Date(Date.now() - (50 - i) * 3600000).toLocaleTimeString('fa-IR');
        }
        return '';
    });
    
    chart.data.labels = labels;
    chart.data.datasets[0].data = sampleData;
    chart.update();
}

// تولید داده‌های نمونه
function generateSampleData(count, basePrice) {
    const data = [];
    let currentPrice = basePrice;
    
    for (let i = 0; i < count; i++) {
        // تغییرات تصادفی کوچک
        const change = (Math.random() - 0.5) * basePrice * 0.02;
        currentPrice += change;
        data.push(currentPrice);
    }
    
    return data;
}

// ایجاد نمودار تحلیل تکنیکال
function createTechnicalChart(canvasId, data) {
    const ctx = document.getElementById(canvasId);
    if (!ctx) return;
    
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.labels || [],
            datasets: [
                {
                    label: 'قیمت',
                    data: data.prices || [],
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    borderWidth: 2,
                    yAxisID: 'y'
                },
                {
                    label: 'میانگین متحرک',
                    data: data.sma || [],
                    borderColor: '#ef4444',
                    borderWidth: 1,
                    borderDash: [5, 5],
                    yAxisID: 'y'
                }
            ]
        },
        options: {
            responsive: true,
            interaction: {
                mode: 'index',
                intersect: false
            },
            scales: {
                x: {
                    display: true,
                    grid: {
                        display: false
                    }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'right'
                }
            },
            plugins: {
                tooltip: {
                    rtl: true,
                    callbacks: {
                        label: function(context) {
                            let label = context.dataset.label || '';
                            if (label) {
                                label += ': ';
                            }
                            if (context.parsed.y !== null) {
                                label += '$' + context.parsed.y.toLocaleString();
                            }
                            return label;
                        }
                    }
                }
            }
        }
    });
}

// به‌روزرسانی داده‌های نمودار در لحظه
function updateRealtimeChart(symbol, price) {
    const chart = charts[symbol.toLowerCase()];
    if (!chart) return;
    
    // اضافه کردن داده جدید
    chart.data.labels.push(new Date().toLocaleTimeString('fa-IR'));
    chart.data.datasets[0].data.push(price);
    
    // حفظ تعداد داده‌ها (50 نقطه آخر)
    if (chart.data.labels.length > 50) {
        chart.data.labels.shift();
        chart.data.datasets[0].data.shift();
    }
    
    chart.update('quiet');
}

// پاک کردن تمام نمودارها
function destroyAllCharts() {
    Object.values(charts).forEach(chart => {
        chart.destroy();
    });
    charts = {};
}
