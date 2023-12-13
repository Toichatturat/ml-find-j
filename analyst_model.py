import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

def model_page():
    st.title("Analyst this Linear Regression Model")
    
    
    df = pd.read_csv("real-salary-edit.csv")
    df = df[["old", "job-title", "salary", "state", "experience", "education", "gender"]].dropna()
    df = df.dropna()
    df = df.rename(columns={"job-title": "job", "old":"age"})
    job_map = shorten_categories(df.job.value_counts(), 100)
    df['job'] = df['job'].map(job_map)
    
    state_map = shorten_categories(df.state.value_counts(), 400)
    df['state'] = df['state'].map(state_map)
    
    df['salary'] = df['salary'].replace('[\$,]', '', regex=True).astype(float)
    
    df = df[df['gender'] != 'Other or prefer not to answer']
    df = df[df['gender'] != 'Prefer not to answer']
    df['gender'] = df['gender'].replace('[\$,]', '', regex=True).astype(float)
    
    le_state = LabelEncoder()
    df['state'] = le_state.fit_transform(df['state'])
    
    le_job = LabelEncoder()
    df['job'] = le_job.fit_transform(df['job'])
    
    df = df[df["salary"] <= 250000]
    df = df[df["salary"] >= 10000]
    df = df[df['state'] != 'Other']
    
    X = df.drop("salary", axis=1)
    y = df["salary"]
    
    linear_reg = LinearRegression()
    linear_reg.fit(X, y.values)
    
    y_pred = linear_reg.predict(X)
    
    corr = df.corr()
    
    st.write("### Correlation")
    
    st.write(corr)
    
    # coefficients = linear_reg.coef_
    # feature_names = X.columns
    
    
    
    st.write("### Coefficients")

    # # แสดงค่า coefficients และ features ที่เกี่ยวข้องเป็นตาราง
    coefficients = linear_reg.coef_
    feature_names = X.columns

    # สร้าง DataFrame
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})
    st.write(coef_df)
        
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    r_squared = r2_score(y, y_pred)
    
    
    
    
    st.write("### Metrics of Model")
    st.write(f'Mean Absolute Error: {mae}')
    st.write(f'Mean Squared Error: {mse}')
    st.write(f'Root Mean Squared Error: {rmse}')
    st.write(f'R-squared: {r_squared}')
    
    st.write("#### Analyzing the metrics of a Linear Regression model is a good step to understanding how your model performs on the data. Here is the preliminary analysis:")
    
    st.write("##### Mean Absolute Error (MAE): The MAE value is 30668.32.")
    st.write("MAE is the average of the difference between the predicted and actual values, without weighting the difference.Here, a high MAE value indicates a large model error.")
    st.write("##### Mean Squared Error (MSE): The MSE value is 1573004062.19.")
    st.write("MSE is the mean of the difference between the predicted and actual values, squared across all items.MSE values are larger than MAE, indicating larger discrepancies at some points.")
    st.write("##### Root Mean Squared Error (RMSE): The RMSE value is 39661.12.")
    st.write("RMSE is the square root of MSE, meaning it has the same dimension as the original variable.A large RMSE value compared to MAE, indicates a large discrepancy.")
    st.write("##### R-squared (R²): The R-squared value is 0.145.")
    st.write("R-squared indicates how well the model can explain changes in the dependent variable. A low R-squared value indicates that the model does not explain the data very well.Improving the model or using a better model should be considered to increase R-squared.")
    
    st.write("### From Correlation will plot that relationships of salary and more information")
    # แสดงค่า coefficients และ features ที่เกี่ยวข้องเป็นตาราง
    st.write(corr)
    
    data = df.groupby(["experience"])["salary"].mean().sort_values(ascending=True)
    # st.line_chart(data)
    
    data2 = df.groupby(["education"])["salary"].mean().sort_values(ascending=True)
    # st.line_chart(data2)
    
    data3 = df.groupby(["age"])["salary"].mean().sort_values(ascending=True)
    # st.line_chart(data3)
    
    data4 = df.groupby(["state"])["salary"].mean().sort_values(ascending=True)
    # st.line_chart(data4)
    
    data5 = df.groupby(["gender"])["salary"].mean().sort_values(ascending=True)
    # st.line_chart(data5)
    
    
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # สร้างกราฟและแสดง title
    st.write("Average Salary by Experience")
    sns.lineplot(data)

    # แสดงกราฟใน Streamlit
    st.pyplot(plt)
    st.write("'5-7 years'=3, '21 - 30 years'=6, '11 - 20 years'=5, '8 - 10 years'=4,'2 - 4 years'=2, '1 year or less'=1,'31 - 40 years'=7,'41 years or more'=8 ")
    
     # สร้างกราฟและแสดง title
    st.write("Average Salary by Education")
    st.line_chart(data2)
    st.write("'Master’s degree'=2, 'Bachelor’s degree'=1, 'Post grad'=3,'Less than a Bachelors'=0 ")
    st.write("Average Salary by Gender")
    st.line_chart(data5)
    
    st.write("Average Salary by Age")
    st.line_chart(data3)
    st.write(" '25-34'=3, '35-44'=4, '45-54'=5, '18-24'=2, '65 or over'=7, '55-64'=6,'under 18'=1 ")