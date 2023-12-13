import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def shorten_categories(categories, cutoff):
    categorical_map = {}
    for i in range(len(categories)):
        if categories.values[i] >= cutoff:
            categorical_map[categories.index[i]] = categories.index[i]
        else:
            categorical_map[categories.index[i]] = 'Other'
    return categorical_map

def clean_experience(x):
    if x ==  'More than 50 years':
        return 50
    if x == 'Less than 1 year':
        return 0.5
    return float(x)


def clean_education(x):
    if 'College degree' in x:
        return 'Bachelor’s degree'
    if "Master's degree" in x:
        return 'Master’s degree'
    if 'Professional degree (MD, JD, etc.)' in x or 'PhD' in x:
        return 'Post grad'
    return 'Less than a Bachelors'




# @st.cache
def load_data():
    df = pd.read_csv("real-salary.csv")
    df = df.rename(columns={"How old are you?": "age", "Job title": "job", "What is your annual salary? (You'll indicate the currency in a later question. If you are part-time or hourly, please enter an annualized equivalent -- what you would earn if you worked the job 40 hours a week, 52 weeks a year.)": "salary","If you're in the U.S., what state do you work in?":"state","What city do you work in?":"city","How many years of professional work experience do you have in your field?":"experience","What is your highest level of education completed?":"education","What is your gender?":"gender"})
    df = df[["age", "job", "salary", "state","city", "experience", "education", "gender"]]
    df = df[["age", "job", "salary", "state","city", "experience", "education", "gender"]].dropna()
    df = df.dropna()
    df['salary'] = df['salary'].replace('[\$,]', '', regex=True).astype(float)
    df = df.drop(columns=['city'])

    country_map = shorten_categories(df.state.value_counts(), 400)
    df["state"] = df["state"].map(country_map)
    df = df[df["salary"] <= 250000]
    df = df[df["salary"] >= 10000]
    df = df[df["state"] != "Other"]
    df = df[df['gender'] != 'Other or prefer not to answer']
    df = df[df['gender'] != 'Non-binary']

    df['education'] = df['education'].apply(clean_education)
    return df

df = load_data()

def show_explore_page():
    st.title("Explore Salaries of USA")

    st.write(
        """
    ### Real Survey from interviewing Survey.
    """
    )
    
    salary_summary = df['salary'].describe()
    st.write(salary_summary)

    data = df["state"].value_counts()

    fig1, ax1 = plt.subplots()
    ax1.pie(data, labels=data.index, autopct="%1.1f%%", shadow=True, startangle=90)
    ax1.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.

    st.write("""#### Number of Data from different countries""")

    st.pyplot(fig1)
    
    st.write(
        """
    #### Mean Salary Based On State
    """
    )

    data = df.groupby(["state"])["salary"].mean().sort_values(ascending=True)
    st.bar_chart(data)

    st.write(
        """
    #### Mean Salary Based On Education
    """
    )

    data = df.groupby(["education"])["salary"].mean().sort_values(ascending=True)
    st.line_chart(data)
    
    st.write(
        """
    #### Mean Salary Based On Experience
    """
    )

    data = df.groupby(["experience"])["salary"].mean().sort_values(ascending=True)
    st.line_chart(data)
    
    