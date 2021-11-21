import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class PredictionModel(object):
    def __init__(self):
        self.day_range = 0
        self.predicted_values = []
        self.max_price = None
        self.min_price = None
        self.max_day = 0
        self.min_day = 0
        self.lr_prediction = 0
        self.svm_prediction = 0

    def __repr__(self):
        return f'Prediction'

    def __len__(self):
        return len(self.predicted_values)

    def predict(self, forecast):
        show_data = False
        self.day_range = forecast  # as forecast duration increases, accuracy decreases
        df = pd.DataFrame(lst, index=['a', 'b', 'c', 'd', 'e', 'f', 'g'], columns =['Names'])
        print(f'Getting data for {self.name}')

        # Setting up the data
        metric = 'Adj. Open'
        df = df[[metric]]
        df['Prediction'] = df[[metric]].shift(-forecast)
        print(f'{metric.title()} Price data: {len(df)} data points')
        x_data = np.array(df.drop(['Prediction'], 1))
        y_data = np.array(df['Prediction'])
        print("Successfully converted the dataframes into arrays.")
        X = x_data[:-forecast]
        Y = y_data[:-forecast]

        # Creating the predictive model
        try:
            # Setting up the training via SVR
            x_tr, x_te, y_tr, y_te = train_test_split(X, Y, test_size=0.2)
            svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
            svr_rbf.fit(x_tr, y_tr)

            # Model confidence scoring
            svm_confidence = svr_rbf.score(x_te, y_te)
            lr = LinearRegression()
            lr.fit(x_tr, y_tr)
            lr_confidence = lr.score(x_te, y_te)
            print(f'SV Model Confidence: {round(svm_confidence * 100, 3)}%')
            print(f'Regression Confidence: {round(lr_confidence * 100, 3)}%')

            # Generate SVR model predictions
            x_forecast = np.array(df.drop(['Prediction'], 1))[-forecast:]
            self.lr_prediction = lr.predict(x_forecast)
            self.svm_prediction = svr_rbf.predict(x_forecast)
            if show_data:
                print(f'SVR Data Prediction:\n{self.svm_prediction}')
                print(f'Linear Regression Data Prediction:\n{self.lr_prediction}')

            # Compiling the data sets
            predicted_values = (self.lr_prediction + self.svm_prediction)
            self.max_price = float(round(np.max(predicted_values), 2))
            self.max_day = int(np.where(predicted_values == np.max(predicted_values))[0])
            self.min_price = float(round(np.min(predicted_values), 2))
            self.min_day = int(np.where(predicted_values == np.min(predicted_values))[0])
            self.predicted_values = predicted_values.tolist()

        except Exception as err:
            print(f'Error during training model: {err}')

    def get_model_data(self):
        price_range = min_max_change(self.min_price, self.max_price, self.current_price)
        print('--- KEY STOCK INFORMATION ---')
        print(f'NAME: {self.name}\nTICKER: {self.ticker}\nPRICE: ${self.current_price}')
        print(f'--- CURRENT DATA FOR THE NEXT {self.day_range} DAYS ---')
        print(f'MAX: ${self.max_price}, in {self.max_day} days')
        print(f'MIN: ${self.min_price}, in {self.min_day} days')
        print(f'PERCENT DIFF. (MAX): {price_range["maximum"]}%')
        print(f'PERCENT DIFF. (MIN): {price_range["minimum"]}%')

    def plot_data(self):
        print('Drawing plot...')
        days = [*range(0, self.day_range, 1)]
        plt.plot(days, self.predicted_values, 'r-', label='Projected Price')
        plt.grid(True)
        plt.xlabel('Days')
        plt.ylabel('Predicted Price')
        plt.title(f'Prediction of {self.name} stock trends for {self.day_range} days')
        plt.legend()
        print(f'{self.name} stock trend plot generated!')
        plt.show()