from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from ydata_profiling import ProfileReport

df = pd.read_csv("studentscores.csv", header=0, sep =",")
x = df.drop('math score', axis=1)
y = df['math score']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
corr = df[['math score','reading score','writing score']].corr()

# profile = ProfileReport(df, title="studentscores Report")
# profile.to_file("studentscores Report.html")

num_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')),
                                  ('scaler', StandardScaler())])

nom_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                  ('encoder', OneHotEncoder())])

education_values = ['some high school', 'high school', 'some college',"associate's degree","bachelor's degree", "master's degree" ]
gender = x_train['gender'].unique()
lunch = x_train['lunch'].unique()
test_preparation = x_train['test preparation course'].unique()
ord_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                  ('encoder', OrdinalEncoder(categories = [education_values,gender,lunch,test_preparation]))])

preprocessor = ColumnTransformer(transformers=[
    ('num_featuers', num_transformer,['reading score','writing score']),
    ('nom_features', nom_transformer, ['race/ethnicity']),
    ('ord_features', ord_transformer, ['parental level of education','gender','lunch','test preparation course'])
])
result = preprocessor.fit_transform(x_train)

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('regressor', LinearRegression())])
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print('MAE:{}'.format(mean_absolute_error(y_test, y_pred)))
print('MSE:{}'.format(mean_squared_error(y_test, y_pred)))
print('R2 :{}'.format(r2_score(y_test, y_pred)))
print(model['regressor'].coef_)