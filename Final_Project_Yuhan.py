#######################
# Import the packages #
#######################
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objs as go

############
# The Tabs #
############
tab1, tab2, tab3, tab4 = st.tabs(["Introduction", "Data EDA", "Training & Testing", "Summary"])

################
# Introduction #
################
with tab1:
    st.markdown("""
    #### CMSE830 Final Project 🎉
    ## ⚽️ 2022 World Cup Prediction ⚽️
    """)

    st.image("960x0.jpg", width=700)
    st.markdown("""
       
    **🎯 The goal of this project is to Predict the champion of this year's World Cup.**

    **📁 What was the dataset?**

    > 1. FIFA World Ranking 1992-2022 Oct 6th
    > 2. Football Results from 1872-2022 Dec 6th

    **What machine learning method was used?**

    'BernoulliNB', 'DecisionTreeClassifier' , **`'GradientBoostingClassifier'`** ,`'GaussianNB'`,
    'LogisticRegression' , 'MLPClassifier',`'RandomForestClassifier'`
    """)

    intro_summary = st.selectbox(label="", options=("","Just Tell Me The Result 🤡"))
    if intro_summary == 'Just Tell Me The Result 🤡':
        st.markdown("""
        Quarter Finals:

        Croatia vs Brazil, Brazil Win

        Netherlands vs Argentina, Netherlands Win

        Morocco vs Portugal, Portugal Win

        England vs France, France Win

        ____

        Semi-finals:

        Brazil vs Netherlands, Brazil Win

        France vs Portugal, Portugal Win

        ____

        3rd:

        Netherlands vs France, Netherlands Win

        ____

        Brazil vs Portugal, Brazil Win


        """)


#############################
# Exploratory Data Analysis #
#############################
with tab2:
    st.header("Exploratory Data Analysis 🥸")

    st.markdown("1. FIFA World Ranking 1992-2022")
    rank = pd.read_csv("fifa_ranking-2022-10-06.csv")
    st.write(rank.head(),rank.shape)

    rank = pd.read_csv("fifa_ranking-2022-10-06.csv")
    rank["rank_date"] = pd.to_datetime(rank["rank_date"])
    rank = rank[(rank["rank_date"] >= "2018-8-1")].reset_index(drop=True)
    rank["country_full"] = rank["country_full"].str.replace("IR Iran", "Iran").str.replace("Korea Republic", "South Korea").str.replace("USA", "United States")

    rank = rank.set_index(['rank_date']).groupby(['country_full'], group_keys=False).resample('D').first().fillna(method='ffill').reset_index()
    ranking_df = pd.pivot_table(data = rank, 
                            values = 'total_points',
                            index = 'country_full',
                            columns = 'rank_date').fillna(0.0)
    
    best_ranks = ranking_df.loc[ranking_df['2018-8-16'].sort_values(ascending = False)[:10].index]
    fig = go.Figure()

    for i in range(len(best_ranks.values)):
        rank_fig = fig.add_trace(go.Scatter(x = best_ranks.columns, 
                                y = best_ranks.iloc[i],
                                name = best_ranks.index[i]))
    
    fig.update_layout(title="FIFA Ranking Evolution for Top 10 Best Teams",yaxis_title="Points")
    st.plotly_chart(rank_fig,use_container_width=True)

    st.markdown("2. Football Results from 1872-2022")
    df =  pd.read_csv("results.csv")
    st.write(df.head(),df.shape)
    df =  pd.read_csv("results.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= "2018-8-1")].reset_index(drop=True)

    st.markdown("3. Combine two datasets ✅ since 2018 last world cup")
    df_merge = df.merge(rank[["country_full", "total_points", "previous_points", "rank", "rank_change", "rank_date"]], left_on=["date", "home_team"], right_on=["rank_date", "country_full"]).drop(["rank_date", "country_full"], axis=1)

    df = df_merge.merge(rank[["country_full", "total_points", "previous_points", "rank", "rank_change", "rank_date"]], left_on=["date", "away_team"], right_on=["rank_date", "country_full"], suffixes=("_home", "_away")).drop(["rank_date", "country_full"], axis=1)
    st.write(df.head(),df.shape)

    heatmap, ax = plt.subplots()
    heatmap.set_size_inches(15, 15)
    corr1 = df.corr()
    sns.heatmap(corr1,annot=True)
    st.write("The Heatmap of Correlation")
    st.pyplot(heatmap)


##########################
# Machine learning model #
##########################
with tab3:
    data = pd.read_csv("model_data.csv")
    st.write(data.describe())
    st.write(data.shape)

    # Actual training start
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import confusion_matrix
    from sklearn.naive_bayes import BernoulliNB, GaussianNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neural_network import MLPClassifier

    X = data.iloc[:, 3:]
    y = data[["target"]]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state=1)

    
##############
# Pick Model #
##############
    st.markdown("""
    ### Which ML Model would you like to have a look?""")

    model = st.selectbox(
    label="", options=('','None of them', 'BernoulliNB', 'DecisionTreeClassifier','GradientBoostingClassifier','GaussianNB',
    'LogisticRegression','MLPClassifier','RandomForestClassifier'))

    if model == 'None of them':
        st.image("17c.jpg", width=700)

    if model == 'BernoulliNB':
        # BernoulliNB
        st.image("BNB.png", width=700)
        st.image("BNB_CV.png", width=700)

    if model == 'DecisionTreeClassifier':
        st.image('DecisionTreeClassifier.png')
        st.image('DecisionTreeClassifier_CV.png')

    if model == 'GradientBoostingClassifier':
        st.image('GradientBoostingClassifier.png')
        st.image('GradientBoostingClassifier_CV.png')

    if model == 'GaussianNB':
        st.image('GaussianNB.png')
        st.image('GaussianNB_CV.png')

    if model == 'LogisticRegression':
        st.image('LogisticRegression.png')
        st.image('LogisticRegression_CV.png')

    if model == 'MLPClassifier':
        st.image('MLPClassifier.png')
        st.image('MLPClassifier_CV.png')

    if model == 'RandomForestClassifier':
        st.image('RandomForestClassifier.png')
        st.image('RandomForestClassifier_CV.png')
        
        
with tab4: 
    st.image('ThisMorning.png')
    summary = st.selectbox(label="", options=("","Here's the prediction"))
    if summary == "Here's the prediction":
        st.markdown("""
        Quarter Finals:

        Croatia vs Brazil, Brazil Win

        Netherlands vs Argentina, Netherlands Win

        Morocco vs Portugal, Portugal Win

        England vs France, France Win

        ____
        Top 4️⃣
        Semi-finals:

        Brazil vs Netherlands, Brazil Win

        France vs Portugal, Portugal Win

        ____
        🥉
        3rd:

        Netherlands vs France, Netherlands Win

        ____
        🏆
        Brazil vs Portugal, Brazil Win


        """)

    st.markdown("""
    #### Things I learned:
    * Cleaning the data is the hardest part. It took me probally 75% of the time.
    * There are tons of classifier and models we can use. 
    * After December 1st, 2020, streamlit remove the ability to specify no arguments in `st.pyplot()` function, which makes it harder to show the plt plot.
    * Soccer is hard to predict.
    
    
    #### Reflection:
    * The real world data is much more complicated than my expectation. I should practice more. 
    * I wish I can add one more selectbox, where we can enter any two soccer team and show the probability of winning.


    #### References:
    * FIFI World Ranking 1992-2022 Oct 6th [data](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017/code?datasetId=4305&sortBy=voteCount)
    * Football Results from 1872-2022 Dec 6th [data](https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017)
    * Data cleaning [here](https://www.kaggle.com/code/sslp23/predicting-fifa-2022-world-cup-with-ml)
    * Picture (Photo by Alexander Hassenstein - FIFA/FIFA via Getty Images)
    

    ## Thank you for listening
    ## Have a wonderful winter break ⛄️
    """)    
        
