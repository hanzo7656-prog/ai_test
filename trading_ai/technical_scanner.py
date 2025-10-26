# اسکنر مبتنی بر معماری اسپارس
class SparseTechnicalScanner:
    def __init__(self, model_path: str = None):
        self.config = SparseConfig()
        if model_path:
            self.model = self.load_model(model_path)
        else:
            self.model = SparseTechnicalNetwork(self.config)
        
    def scan_market(self, symbols: List[str], conditions: Dict) -> List[Dict]:
        """اسکن بازار با شرایط تکنیکال"""
        results = []
        
        for symbol in symbols:
            # دریافت داده‌های بازار
            market_data = self.get_market_data(symbol)
            
            # تحلیل با مدل اسپارس
            analysis = self.model(market_data)
            
            # بررسی شرایط
            if self.check_conditions(analysis, conditions):
                results.append({
                    'symbol': symbol,
                    'analysis': analysis,
                    'timestamp': np.datetime64('now')
                })
        
        return results
    
    def get_technical_recommendations(self, analysis: Dict) -> List[str]:
        """تولید توصیه‌های تحلیلی"""
        recommendations = []
        
        trend = torch.softmax(analysis['trend_strength'], dim=-1)
        if trend[0] > 0.6:  # صعودی قوی
            recommendations.append("روند صعودی قوی")
        elif trend[1] > 0.6:  # نزولی قوی
            recommendations.append("روند نزولی قوی")
            
        confidence = analysis['overall_confidence'].item()
        if confidence > 0.7:
            recommendations.append("سیگنال با اطمینان بالا")
            
        return recommendations
