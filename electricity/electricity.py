import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from lazypredict.Supervised import LazyRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv('electricity.csv')
# profile = ProfileReport(data)
# profile.to_file('report.html')

# fig, ax = plt.subplots()
# ax.plot(data['Time'], data['Demand'], label='Demand')
# plt.xticks([0,25000,50000])
# plt.show()

def convert_data(data_raw, windows):
    i=1
    while i < windows:
        data_raw['Demand_t-{}'.format(i)] = data_raw['Demand'].shift(i)
        i+=1
    data_raw = data_raw.dropna(axis=0)
    return data_raw

data['Period'] = (pd.to_datetime(data['Time']).dt.hour * 2 + pd.to_datetime(data['Time']).dt.minute//30)
#tức là time từ file là chuỗi, cần chuyển sang dạng datetime; dt.hour là trình truy cập thuộc tính dùng để truy cập lấy giờ trong datetime (trong timestamps k đc)
data = convert_data(data,3)
x = data.drop(['Demand','Time'], axis=1)
y = data['Demand']
x_train,x_test, y_train,y_test = train_test_split(x,y, test_size = 0.2, shuffle = False)

preprocessor = ColumnTransformer(transformers=[
        ('Onehot', OneHotEncoder(handle_unknown='ignore'), ['Holiday']),
        ('scaler', StandardScaler(), ['Temperature','Demand_t-1','Demand_t-2']),
        ])

selected_models = [('LinearRegression', LinearRegression())]
reg = LazyRegressor(
    verbose = 0,
    ignore_warnings=True,
    custom_metric= None ,
    predictions=False,
    regressors =selected_models
)
models, predictions = reg.fit(x_train,x_test,y_train,y_test)

# model = Pipeline(steps=[
#     ('preprocessor', preprocessor),
#     ('regressor', RandomForestRegressor(max_depth=10,min_samples_leaf=50,n_estimators=100)),
# ])
# model.fit(x_train,y_train)
# y_predict = model.predict(x_test)
#
# print('MSE:{}'.format(mean_squared_error(y_test,y_predict)))
# print('MAE:{}'.format(mean_absolute_error(y_test,y_predict)))
# print('R2:{}'.format(r2_score(y_test,y_predict)))

#trực quan hóa có cả predict -> lưu mô hình-> tạo script mới xuất mô hình - > thử nghiệm dữ liệu của khách