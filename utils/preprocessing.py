import pandas as pd
from sklearn.preprocessing import StandardScaler



class Preprocessor ():
    def __init__(self, max_train_len=180, test_len=30, split_coeff=0.1):
        self.train_len = max_train_len
        self.test_len = test_len
        self.split = split_coeff

        return
    


    def __call__(self, history):
        """
        При вызове генерирует признаки для истории и возвращает разделённую на train/val выборку
        """
        # Возвращает pd.DataFrame с полным набором фичей (который можно при необходимости расширить)
        p_history = self.process_history(history)

        window_data = None
        columns_to_normalize = ['Close', 'Open', 'High', 'Low', 'EMA 9', 'EMA 20', 'EMA 50', 'EMA 200']

        splitted_history = []
        window = (self.train_len + self.test_len)
        for i in range(0, len(p_history) - window + 1):
            window_data = p_history.iloc[i:i+window].copy()
            
            # Нормирование выбранных столбцов на первую цену закрытия
            first_close = window_data.iloc[0]['Close']
            window_data[columns_to_normalize] /= first_close
            window_data[columns_to_normalize] -= 1

            splitted_history.append(window_data)

        test_len = int(len(splitted_history) * self.split)
        train, val = splitted_history[:-test_len], splitted_history[-test_len:]

        return train, val

    
    
    def process_history (self, history):
        """
        Данная функция добавляет столбцы со значениями новых фичей в переданный pd.DataFrame history
        в нормализованном виде
        """        
        # Нормализуем объём
        history['Volume'] = StandardScaler().fit_transform(history[['Volume']])
        
        # Добавление EMA на нескольких периодах (9, 20, 50, 200)
        history['EMA 9'] = self._ema_(history['Close'], window=9)
        history['EMA 20'] = self._ema_(history['Close'], window=20)
        history['EMA 50'] = self._ema_(history['Close'], window=50)
        history['EMA 200'] = self._ema_(history['Close'], window=200)

        # Добавление волатильности на нескольких периодах (9, 20, 50, 200)
        history['Volatility 9'] = self._volatility_(history['Close'], window=9) / history['EMA 9']
        history['Volatility 20'] = self._volatility_(history['Close'], window=20) / history['EMA 20']
        history['Volatility 50'] = self._volatility_(history['Close'], window=50) / history['EMA 50']
        history['Volatility 200'] = self._volatility_(history['Close'], window=200) / history['EMA 200']

        # Добавление RSI
        history['RSI 5'] = self._rsi_(history['Close'], window=5)/100 - 0.5
        history['RSI 9'] = self._rsi_(history['Close'], window=9)/100 - 0.5
        history['RSI 14'] = self._rsi_(history['Close'], window=14)/100 - 0.5
        
        # Добавление MACD (со cтандартными параметрами 12/26/9)
        history['MACD line'], history['MACD signal'] = self._macd_(history['Close'], fast=12, slow=26, window=9)

        # Добавление Stochastic (с популярными значениями окон )
        history["Stochastic %D 5"] = self._stochastic_oscillator_(history['Close'], window=5, smooth=3)/100 - 0.5
        history["Stochastic %D 9"] = self._stochastic_oscillator_(history['Close'], window=9, smooth=3)/100 - 0.5
        history["Stochastic %D 14"] = self._stochastic_oscillator_(history['Close'], window=14, smooth=3)/100 - 0.5

        # Добавление RVI (а он из коробки нормирован)
        history['RVI 5'] = self._rvi_(history['Open'], history['High'], history['Low'], history['Close'], window=5) 
        history['RVI 9'] = self._rvi_(history['Open'], history['High'], history['Low'], history['Close'], window=9) 
        history['RVI 14'] = self._rvi_(history['Open'], history['High'], history['Low'], history['Close'], window=14) 

        # Добавление MFI
        history['MFI'] = self._mfi_(history['High'], history['Low'], history['Close'], history['Volume'], window=14)/100 - 0.5

        # Добавление Awsome Oscillator
        history['AO'] = self._awesome_oscillator_(history['Close']) / history['EMA 20']

        # При необходимости сюда можно добавить ещё осцилляторов
        #   CODE HERE <...>

        # Подрезаем на 30 (число с потолка) историю и убираем NaN
        history = history.iloc[30:].fillna(0)

        # Возвращаем историю в более легковесном формате 'float32'
        return history.astype('float32')
    

    
    def _ema_(self, prices, window=5):
        """
        Расчитывает EMA для цен закрытия бумаги
        """
        ema = prices.ewm(span=window, adjust=False).mean()
        
        return ema
    


    def _volatility_(self, prices, window=14):
        """
        Расчитывает волатильность на основе стандартного отклонения цен закрытия
        """
        volatility = prices.rolling(window=window, min_periods=1).std()
        
        return volatility
    
    

    def _rsi_(self, prices, window=14):
        """
        Расчитывает индикатор RSI для цен закрытия бумаги
        """
        # Разница между последовательными ценами 
        delta = prices.diff().dropna()

        # Отрицательные и положительные изменения
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Вычисление скользящего среднего для приростов и убытков
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()

        # Вычисление отношения средних приростов и убытков
        rs = avg_gain / avg_loss

        # Расчет RSI
        rsi = 100 - (100 / (1 + rs))

        return rsi
    
    

    def _macd_(self, prices, fast=12, slow=26, window=9):
        """
        Расчитывает индикатор MACD для цен закрытия бумаги
        """
        # Вычисление быстрой и медленной скользящих средних
        ema_fast = prices.ewm(span=fast, min_periods=1, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, min_periods=1, adjust=False).mean()

        # Расчет MACD линии и сигнальной линии
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=window, min_periods=1, adjust=False).mean()

        return macd_line, signal_line
    
    

    def _stochastic_oscillator_(self, prices, window=14, smooth=3):
        """
        Расчитывает индикатор Stochastic Oscillator для цен закрытия бумаги
        """
        # Расчет минимальных и максимальных цен за период
        min_prices = prices.rolling(window=window, min_periods=1).min()
        max_prices = prices.rolling(window=window, min_periods=1).max()

        # Вычисление %K
        percent_k = ((prices - min_prices) / (max_prices - min_prices)) * 100

        # Сглаживание %K для получения %D (сигнальной линии)
        percent_d = percent_k.rolling(window=smooth, min_periods=1).mean()

        return percent_d
    

    
    def _rvi_(self, open, high, low, close, window=14):
        """
        Расчитывает индикатор RVI 
        """
        a = close - open
        b = 2 * (close.shift(1) - open.shift(1))
        c = 2 * (close.shift(2) - open.shift(2))
        d = close.shift(3) - open.shift(3)
        numerator = (a + b + c + d) / 6
        
        e = high - low
        f = 2 * (high.shift(1) - low.shift(1))
        g = 2 * (high.shift(2) - low.shift(2))
        h = high.shift(3) - low.shift(3)
        denominator = (e + f + g + h) / 6
        
        numerator_sum = numerator.rolling(4).sum()
        denominator_sum = denominator.rolling(4).sum()

        rvi = (numerator_sum / denominator_sum).rolling(window).mean()
        rvi1 = 2 * rvi.shift(1)
        rvi2 = 2 * rvi.shift(2)
        rvi3 = rvi.shift(3)

        rvi_signal = (rvi + rvi1 + rvi2 + rvi3) / 6
        
        return rvi_signal
    

    
    def _mfi_(self, high, low, close, volume, window=14):
        """
        Расчитывает индикатор Money Flow Index (MFI)
        """
        # Расчет типичной цены
        typical_price = (high + low + close) / 3

        # Расчет денежного потока
        money_flow = typical_price * volume

        # Расчет изменения денежного потока
        money_flow_change = money_flow.diff()

        # Расчет положительного и отрицательного денежного потока
        positive_flow = money_flow_change.where(money_flow_change > 0, 0)
        negative_flow = -money_flow_change.where(money_flow_change < 0, 0)

        # Расчет суммы положительного и отрицательного денежного потока за период
        positive_flow_sum = positive_flow.rolling(window=window, min_periods=1).sum()
        negative_flow_sum = negative_flow.rolling(window=window, min_periods=1).sum()

        # Расчет отношения позитивного и негативного денежного потока
        money_flow_ratio = positive_flow_sum / negative_flow_sum

        # Расчет индекса MFI
        mfi = 100 - (100 / (1 + money_flow_ratio))

        return mfi
    


    def _awesome_oscillator_(self, close, fast=5, slow=34):
        """
        Расчитывает индикатор Awesome Oscillator (AO)
        """
        # Вычисление разницы между скользящими средними
        ao = close.rolling(window=fast, min_periods=1).mean() - close.rolling(window=slow, min_periods=1).mean()

        return ao