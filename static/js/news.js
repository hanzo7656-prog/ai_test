// مدیریت صفحه اخبار
let currentNewsType = 'all';
let currentNewsPage = 1;
const newsPerPage = 9;

document.addEventListener('DOMContentLoaded', function() {
    loadNews('all');
    setupNewsEventListeners();
});

// بارگذاری اخبار
async function loadNews(type = 'all', page = 1) {
    try {
        showNewsSkeleton();
        updateNewsFilters(type);
        
        let endpoint = '/news';
        if (type !== 'all') {
            endpoint = `/news/${type}`;
        }
        
        const data = await apiCall(`${endpoint}?limit=${newsPerPage}`);
        
        if (data && data.result) {
            renderNewsCards(data.result);
            currentNewsType = type;
            currentNewsPage = page;
        }
    } catch (error) {
        handleError(error, 'بارگذاری اخبار');
    }
}

// رندر کارت‌های خبر
function renderNewsCards(news) {
    const container = document.getElementById('news-container');
    const skeleton = document.getElementById('news-skeleton');
    
    skeleton.classList.add('hidden');
    
    if (!news || news.length === 0) {
        container.innerHTML = `
            <div class="col-span-3 text-center py-8 text-gray-500">
                <div class="text-4xl mb-4">📰</div>
                <p>خبری یافت نشد</p>
            </div>
        `;
        return;
    }
    
    container.innerHTML = news.map(item => `
        <div class="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow duration-200">
            ${item.img ? `
            <img src="${item.img}" alt="${item.title}" class="w-full h-48 object-cover">
            ` : `
            <div class="w-full h-48 bg-gradient-to-r from-blue-400 to-purple-500 flex items-center justify-center">
                <span class="text-white text-4xl">📰</span>
            </div>
            `}
            
            <div class="p-6">
                <div class="flex justify-between items-start mb-3">
                    <span class="bg-blue-100 text-blue-600 px-2 py-1 rounded text-xs font-semibold">
                        ${item.source || 'منبع ناشناس'}
                    </span>
                    <span class="text-gray-500 text-sm">
                        ${formatNewsDate(item.feedDate || item.date)}
                    </span>
                </div>
                
                <h3 class="font-semibold text-lg mb-3 line-clamp-2">
                    ${item.title || 'عنوان نامعلوم'}
                </h3>
                
                <p class="text-gray-600 text-sm mb-4 line-clamp-3">
                    ${item.description || item.content || 'توضیحات در دسترس نیست'}
                </p>
                
                <div class="flex justify-between items-center">
                    <span class="text-xs text-gray-500">
                        ${getSentimentBadge(item.sentiment)}
                    </span>
                    <a href="${item.link || item.url || '#'}" target="_blank" 
                       class="bg-blue-600 text-white px-4 py-2 rounded text-sm hover:bg-blue-700 transition-colors">
                        مطالعه کامل
                    </a>
                </div>
            </div>
        </div>
    `).join('');
}

// نمایش نشان احساسات خبر
function getSentimentBadge(sentiment) {
    const sentiments = {
        'bullish': { text: 'صعودی', class: 'bg-green-100 text-green-800' },
        'bearish': { text: 'نزولی', class: 'bg-red-100 text-red-800' },
        'neutral': { text: 'خنثی', class: 'bg-gray-100 text-gray-800' }
    };
    
    const sentimentData = sentiments[sentiment] || sentiments.neutral;
    
    return `<span class="px-2 py-1 rounded text-xs font-semibold ${sentimentData.class}">
        ${sentimentData.text}
    </span>`;
}

// به‌روزرسانی فیلترهای فعال
function updateNewsFilters(activeType) {
    document.querySelectorAll('.news-filter').forEach(btn => {
        if (btn.textContent.includes(activeType) || 
            (activeType === 'all' && btn.textContent.includes('همه'))) {
            btn.classList.remove('bg-gray-200', 'text-gray-700');
            btn.classList.add('bg-blue-600', 'text-white');
        } else {
            btn.classList.remove('bg-blue-600', 'text-white');
            btn.classList.add('bg-gray-200', 'text-gray-700');
        }
    });
}

// نمایش اسکلت اخبار
function showNewsSkeleton() {
    const container = document.getElementById('news-container');
    const skeleton = document.getElementById('news-skeleton');
    
    container.innerHTML = '';
    skeleton.classList.remove('hidden');
}

// تنظیم event listeners
function setupNewsEventListeners() {
    // بارگذاری اخبار بیشتر
    document.getElementById('load-more-news').addEventListener('click', function() {
        currentNewsPage++;
        loadMoreNews();
    });
}

// بارگذاری اخبار بیشتر
async function loadMoreNews() {
    try {
        let endpoint = '/news';
        if (currentNewsType !== 'all') {
            endpoint = `/news/${currentNewsType}`;
        }
        
        const data = await apiCall(`${endpoint}?limit=${newsPerPage}&offset=${(currentNewsPage - 1) * newsPerPage}`);
        
        if (data && data.result && data.result.length > 0) {
            appendNewsCards(data.result);
        } else {
            showNotification('اخبار بیشتری برای نمایش وجود ندارد', 'info');
            currentNewsPage--;
        }
    } catch (error) {
        handleError(error, 'بارگذاری اخبار بیشتر');
        currentNewsPage--;
    }
}

// اضافه کردن کارت‌های خبر جدید
function appendNewsCards(news) {
    const container = document.getElementById('news-container');
    
    const newCards = news.map(item => `
        <div class="bg-white rounded-lg shadow-md overflow-hidden hover:shadow-lg transition-shadow duration-200">
            ${item.img ? `
            <img src="${item.img}" alt="${item.title}" class="w-full h-48 object-cover">
            ` : `
            <div class="w-full h-48 bg-gradient-to-r from-blue-400 to-purple-500 flex items-center justify-center">
                <span class="text-white text-4xl">📰</span>
            </div>
            `}
            
            <div class="p-6">
                <div class="flex justify-between items-start mb-3">
                    <span class="bg-blue-100 text-blue-600 px-2 py-1 rounded text-xs font-semibold">
                        ${item.source || 'منبع ناشناس'}
                    </span>
                    <span class="text-gray-500 text-sm">
                        ${formatNewsDate(item.feedDate || item.date)}
                    </span>
                </div>
                
                <h3 class="font-semibold text-lg mb-3 line-clamp-2">
                    ${item.title || 'عنوان نامعلوم'}
                </h3>
                
                <p class="text-gray-600 text-sm mb-4 line-clamp-3">
                    ${item.description || item.content || 'توضیحات در دسترس نیست'}
                </p>
                
                <div class="flex justify-between items-center">
                    <span class="text-xs text-gray-500">
                        ${getSentimentBadge(item.sentiment)}
                    </span>
                    <a href="${item.link || item.url || '#'}" target="_blank" 
                       class="bg-blue-600 text-white px-4 py-2 rounded text-sm hover:bg-blue-700 transition-colors">
                        مطالعه کامل
                    </a>
                </div>
            </div>
        </div>
    `).join('');
    
    container.innerHTML += newCards;
}

// فرمت تاریخ خبر
function formatNewsDate(dateString) {
    if (!dateString) return 'تاریخ نامعلوم';
    
    try {
        const date = new Date(dateString);
        return new Intl.DateTimeFormat('fa-IR', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        }).format(date);
    } catch (error) {
        return dateString;
    }
}
