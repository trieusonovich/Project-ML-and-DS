import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score

data = pd.read_csv('web-traffic new data.csv',header = 0)
with open('web traffic model.pkl','rb') as f:
    model = pickle.load(f)

def convert_data(raw_data, windows):
    i = 1
    while i < windows:
        raw_data['users_t-{}'.format(i)] = raw_data['users'].shift(i)
        i+= 1
    raw_data = raw_data.dropna()
    return raw_data

data= convert_data(data, 10)
x_test = data.drop(['date','users'],axis=1)
y_predict= model.predict(x_test)

for i,j in zip(y_predict, data['users']):
    print('y dự đoán: {},y thực tế:{}'.format(i,j))

print('MAE:{}'.format(mean_absolute_error(data['users'],y_predict)))
print('MSE:{}'.format(mean_squared_error(data['users'],y_predict)))
print('R2:{}'.format(r2_score(data['users'],y_predict)))

fig,ax= plt.subplots()
ax.plot(data['date'],data['users'], label = 'thực tế')
ax.plot(data['date'],y_predict, label = 'dự đoán')
plt.xticks([0,20,40,60,80], rotation = 45)
plt.legend()
plt.show()