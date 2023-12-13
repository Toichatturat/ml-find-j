import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_model():
    with open('saved_steps.pkl', 'rb') as file:
        data = pickle.load(file)
    return data

data = load_model()

regressor_loaded = data["model"]
le_age = data["le_age"]
le_job = data["le_job"]
le_state = data["le_state"]
le_experience = data["le_experience"]
le_education = data["le_education"]
le_gender = data["le_gender"]

def show_predict_page():
    st.title("Salary Prediction from real data")

    st.write("It’s hard to get real-world information about what jobs pay, ALISON GREEN published a survey in 2021 on AskAManager.org, a US-centric-ish but does allow for a range of country inputs. The survey is designed to examine payment of different industries based on experience years, field experience years among other variables such as gender, race and education level.")
    st.write("The dataset is “live” and constantly growing, our dataset was downloaded in 23/2/2023.")
    st.write("Data from surveys and inquiries of more than twenty thousand people.")
    st.write("##### Thanks for Data from")
    st.markdown("[https://https://www.kaggle.com/datasets/20c91a0912cae7984d05983e87f68b9710996f40fcc4a2e14a4ec54285da8db8)](https://https://www.kaggle.com/datasets/20c91a0912cae7984d05983e87f68b9710996f40fcc4a2e14a4ec54285da8db8)")
    
    st.write("""### We need some information to predict the salary""")

    age = (
        '18-24','25-34','35-44','45-54','55-64','65 or over','under 18',
    )
    
    job = (
        'Director', 'Executive Assistant', 'Librarian' ,'Manager' ,'Other',
        'Program Manager', 'Project Manager', 'Senior Software Engineer',
        'Software Engineer' ,'Teacher'
    )
    
    state = (
        'California' ,'Colorado' ,'District of Columbia' ,'Florida', 'Georgia',
        'Illinois', 'Maryland' ,'Massachusetts', 'Michigan' ,'Minnesota','New York',
        'North Carolina', 'Ohio' ,'Oregon', 'Pennsylvania', 'Texas', 'Virginia',
        'Washington', 'Wisconsin'
    )
    
    experience = (
        '1 year or less', '11 - 20 years', '2 - 4 years' ,'21 - 30 years',
        '31 - 40 years' ,'41 years or more', '5-7 years', '8 - 10 years'
    )
    
    education = (
        "Bachelor’s degree" ,"Less than a Bachelors", "Master’s degree", "Post grad"
    )
    
    gender = (
        'Man' ,'Woman'
    )
    
    age = st.selectbox("Age", age)
    job = st.selectbox("Job Title", job)
    state = st.selectbox("State", state)
    experience = st.selectbox("Experience", experience)
    education = st.selectbox("Education", education)
    gender = st.selectbox("Gender", gender)
    
    ok = st.button("Calculate Salary")
    if ok:
        X = np.array([[age, job, state,experience,education, gender]])
        X[:, 0] = le_age.transform(X[:,0])
        X[:, 1] = le_job.transform(X[:,1])
        X[:, 2] = le_state.transform(X[:,2])
        X[:, 3] = le_experience.transform(X[:,3])
        X[:, 4] = le_education.transform(X[:,4])
        X[:, 5] = le_gender.transform(X[:,5])
        X = X.astype(float)

        salary = regressor_loaded.predict(X)
        st.subheader(f"The estimated salary is ${salary[0]:.2f} USD")
        
    st.title(f"From model this bar chart to show features importance for Salary")
        
    # Get feature importances from the model
    headers = ["name", "score"]
    values = sorted(zip(["Age", "Job Title", "State", "Experience", "Education", "Gender"], regressor_loaded.feature_importances_), key=lambda x: x[1] * -1)
    rf_feature_importances = pd.DataFrame(values, columns=headers)

    # Plot feature importances
    fig = plt.figure(figsize=(15, 7))
    x_pos = np.arange(0, len(rf_feature_importances))
    plt.bar(x_pos, rf_feature_importances['score'])
    plt.xticks(x_pos, rf_feature_importances['name'])
    plt.xticks(rotation=90)
    plt.title('Feature importances for Salary')

    st.pyplot(fig)  # แสดงกราฟใน Streamlit