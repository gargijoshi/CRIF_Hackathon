import streamlit as st
import requests
from datetime import date
from api import API_KEY
import spacy
from textblob import TextBlob
import pandas as pd
import numpy as np
import json
import spacy_streamlit
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

def wrdcld(Negative_sent):
    ls=Negative_sent.split()
    s=set()
    with open('negative-words.txt') as file:
        contents = file.read()
        for search_word in ls:
            search_word1="\n"+search_word+"\n"
            if search_word1 in contents:
                s.add(search_word)
    
    wordcloud = WordCloud().generate(" ".join(s))
    # Display the generated image:
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()


st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(layout="wide")

#pip freeze > requirements.txt

st.title("Media Analytics")
tabs = st.tabs(["Media", "Dashboard", "Compare"])

sid_obj = SentimentIntensityAnalyzer()

count1, count2 = 0, 0

tab_media = tabs[0]
with tab_media:
    # st.write("Media")
    # st.title("Media Analytics")
    data_df = {
                "Title" : [],
                "Description" : [],
                "Content" : [],
                "Polarity" : [],
                "Subjectivity" : []
    }

    data_df_compare = {
                "Title" : [],
                "Description" : [],
                "Content" : [],
                "Polarity" : [],
                "Subjectivity" : []
    }


    company = st.text_input("Enter Company Name:")

    col1, col2 = st.columns([2,2])
    with col1:
        from_date = st.date_input("From: ", max_value=date.today())
        to_date = st.date_input("To: ", min_value=from_date, max_value=date.today())
    with col2:
        sort = st.radio("Sort By: ", ("Latest", "Popularity", "Relevancy"))
        if sort == "Latest":
            sort = "publishedAt"

        N = st.slider('Maximum results to fetch:', 0, 100, 20)
    company2 = st.text_input("Enter another company for comparison : ")
    btn = st.button("Enter")

    if company != "":
        st.sidebar.title(company)
        st.sidebar.markdown(""+str(from_date)+" to "+str(to_date))

    nlp = spacy.load("en_core_web_sm")

    # btn1 = st.button("Dashboard")
    placeholder = st.empty()

    if btn:
        if company == "":
            st.warning('Please Enter Company Name', icon="⚠️")
        else:
            url = f"https://newsapi.org/v2/everything?q={company}&from={from_date}&to={to_date}&language=en&sortBy={sort}&apiKey={API_KEY}"
            r = requests.get(url).json()

            if(r["totalResults"]==0):
                st.warning('Please enter another Company Name', icon="⚠️")
                print("Error")
            else:
                articles = r["articles"][:N]
                

                Positive_sent=""
                Negative_sent = ""
                Neutral_sent= ""

                for article in articles:
                    title = str(article["title"])
                    desc = str(article["description"])
                    content = str(article["content"])

                    # st.header(title)
                    # st.write(article["publishedAt"])
                    # st.write(article["source"]["name"])
                    # st.write(desc)
                    data = title + " " + desc + " " + content

                    data_df["Title"].append(title)
                    data_df["Description"].append(desc)
                    data_df["Content"].append(content)
                
                    #sentiment analysis
                    blob = TextBlob(data)
                    result_sentiment = blob.sentiment
                    #st.success(result_sentiment)
                    data_df["Polarity"].append(result_sentiment[0])
                    data_df["Subjectivity"].append(result_sentiment[1])

                    if(result_sentiment[0]>0):
                        Positive_sent+=title+" "+desc+" "+content+" "
                    elif(result_sentiment[0]<0):
                        Negative_sent+=title+" "+desc+" "+content+" "
                    else:
                        Neutral_sent+=title+" "+desc+" "+content+" "

                

                ls=Negative_sent.split()
                s=set()
                with open('negative-words.txt') as file:
                    contents = file.read()
                    for search_word in ls:
                        search_word1="\n"+search_word+"\n"
                        if search_word1 in contents:
                            s.add(search_word)
                
                if(len(s)!=0):
                    wordcloud = WordCloud().generate(" ".join(s))
                    # Display the generated image:
                    plt.imshow(wordcloud, interpolation='bilinear')
                    plt.axis("off")
                    plt.show()
                    st.sidebar.markdown("__________________________________________")
                    st.sidebar.title("Reputational Risk Element")           
                    st.sidebar.pyplot()
                    # st.write(s)

                    
                temp_positive = pd.DataFrame(Counter(Positive_sent.split()).most_common())
                temp_positive.columns = ['Common_words','count']
                # st.dataframe(temp_positive.style.background_gradient(cmap='Greens'))


                # st.dataframe(pd.DataFrame(data_df))
                docx = data_df
                data_df = pd.DataFrame(data_df)
                N = data_df.shape[0]
                st.write(N," Results")
                temp_df = data_df[["Title","Polarity"]].sort_values("Polarity")
                st.dataframe(temp_df.style.background_gradient(cmap='RdYlGn'))

                docx = nlp(json.dumps(docx))
                
                companies = set()
                for token in docx.ents:
                    if token.label_ == 'ORG': 
                        if token.text[0] == '+':
                            continue
                        companies.add(token.text)               

                company_avg = {}

                for comp in companies:
                    for row in data_df[["Title", "Description", "Content", "Polarity"]].itertuples(index=False):
                        row_data = row[0] +" "+ row[1] +" "+ row[2]
                        rowx = nlp(row_data)
                        for token in rowx.ents:
                            if token.label_ == 'ORG' and token.text == comp:
                                if comp not in company_avg.keys():
                                    if row[3] > 0:
                                        company_avg[comp] = {"Pos": row[3], "pos_count": 1, "Neg": 0, "neg_count": 0}
                                    if row[3] < 0:
                                        company_avg[comp] = {"Pos": 0, "pos_count": 0, "Neg": row[3], "neg_count": 1}
                                else:
                                    if row[3] > 0:
                                        company_avg[comp] = {"Pos": company_avg[comp]["Pos"]+row[3], "pos_count": company_avg[comp]["pos_count"]+1, "Neg": company_avg[comp]["Neg"], "neg_count": company_avg[comp]["neg_count"]}
                                    if row[3] < 0:
                                        company_avg[comp] = {"Pos": company_avg[comp]["Pos"], "pos_count": company_avg[comp]["pos_count"], "Neg": company_avg[comp]["Neg"]+row[3], "neg_count": company_avg[comp]["neg_count"]+1}

                company_avg_df = {
                            "Company" : [],
                            "Average Positive Sentiment" : [],
                            "Average Negative Sentiment" : []
                        }

                for comp in company_avg.keys():
                    company_avg_df["Company"].append(comp)
                    if company_avg[comp]["pos_count"] != 0:
                        company_avg_df["Average Positive Sentiment"].append(company_avg[comp]["Pos"] / company_avg[comp]["pos_count"])
                    if company_avg[comp]["pos_count"] == 0:
                        company_avg_df["Average Positive Sentiment"].append(0)
                    if company_avg[comp]["neg_count"] != 0:
                        company_avg_df["Average Negative Sentiment"].append(company_avg[comp]["Neg"] / company_avg[comp]["neg_count"])
                    if company_avg[comp]["neg_count"] == 0:
                        company_avg_df["Average Negative Sentiment"].append(0)

              #Dashboard tab
                tab_dashboard = tabs[1]
                with tab_dashboard:
                    dash_col1, dash_col2 = st.columns([3,2])
                    negatives = 0
                    positive = 0
                    neutral = 0
                    
                    for i in range(0,N):
                        if data_df["Polarity"][i] < 0:
                            negatives = negatives + 1
                        if data_df["Polarity"][i] > 0:
                            positive = positive + 1
                        if data_df["Polarity"][i] == 0:
                            neutral = neutral + 1 

                    labels = ["Negative", "Positive", "Neutral"]
                    fig = px.pie(labels, values = [negatives, positive, neutral], hole = 0.3,
                    names = labels, color = labels,
                        width=300, height=300,
                    color_discrete_map = {'Negative':'red', 
                                            'Positive': 'green',
                                    'Neutral':'seablue'
                    })
                    with dash_col1: 
                        st.subheader("Polarity & Subjectivity")
                        chart_data = pd.DataFrame({"Polarity":data_df["Polarity"], "Subjectivity":data_df["Subjectivity"]})    
                        st.line_chart(chart_data)

                    with dash_col2:
                        st.subheader("Polarity Distribution")
                        st.plotly_chart(fig)    

                    st.subheader("Company-wise Average Sentiment")
                    fig = px.bar(company_avg_df, x='Company', y=['Average Positive Sentiment','Average Negative Sentiment'], color_discrete_map={'Average Positive Sentiment':'green', 'Average Negative Sentiment':'red'})
                    st.plotly_chart(fig)

    #Comparison tab
    tab_compare = tabs[2]
    with tab_compare:
        if company2 == "":
            st.write('Company not specified')        
        else:
            st.write("Comparing " + company2 + " with " + company)
            url1 = f"https://newsapi.org/v2/everything?q={company2}&from={from_date}&to={to_date}&language=en&sortBy={sort}&apiKey={API_KEY}"
            r1 = requests.get(url1).json()

            if(r1["totalResults"]==0):
                st.warning('Please enter correct Company Name', icon="⚠️")
                print("Error")
            else:
                articles1 = r1["articles"][:N]
                st.write(N," Results")

                for article in articles1:
                    title1 = str(article["title"])
                    desc1 = str(article["description"])
                    content1 = str(article["content"])

                    data1 = title1 + " " + desc1 + " " + content1

                    data_df_compare["Title"].append(title1)
                    data_df_compare["Description"].append(desc1)
                    data_df_compare["Content"].append(content1)
                
                    #sentiment analysis
                    blob1 = TextBlob(data1)
                    result_sentiment1 = blob1.sentiment
                    data_df_compare["Polarity"].append(result_sentiment1[0])
                    data_df_compare["Subjectivity"].append(result_sentiment1[1])
            
                chart_data = pd.DataFrame({company+" Polarity":data_df["Polarity"][:len(data_df_compare["Polarity"])], company2+" Polarity":data_df_compare["Polarity"]})    
                st.line_chart(chart_data)
 
  