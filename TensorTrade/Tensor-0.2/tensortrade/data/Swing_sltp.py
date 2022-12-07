
def calculate_sltp(self, current_index, action, fill_price):
    
	minimum_risk = self.sl_value
    if self.sl_mode == 'fixed':  # fix mode
        self.risk = self.sl_value
        self.stop_loss = fill_price - action * self.risk
        self.take_profit = fill_price + action * self.risk 

    else: #Swing mode
        if action == 1:
            low_data = self.data_dict['low'][:current_index + 1]
            swing_low_indices = argrelextrema(low_data, lambda x, y: x < y, order=self.scipy_neighborhood, )[0]
            try:
                self.stop_loss = low_data[next(x for x in swing_low_indices[::-1] if low_data[x] < fill_price)]
            except:
                self.stop_loss = fill_price - minimum_risk
            self.risk = abs(self.stop_loss - fill_price)
            self.take_profit = fill_price + self.risk * self.R2R

        else:
            high_data = self.data_dict['high'][:current_index+1]
            swing_high_indices = argrelextrema(high_data, lambda x, y: x > y, order=self.scipy_neighborhood)[0]
            try:
                self.stop_loss =  high_data[next(x for x in swing_high_indices[::-1] if high_data[x] > fill_price)]
            except:
                self.stop_loss = fill_price + minimum_risk
            self.risk = abs(self.stop_loss - fill_price)
            self.take_profit = fill_price - self.risk * self.R2R
