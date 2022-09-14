#Import Packages
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import datasets
import plotly.tools

#Load file
file = "mall_customer.csv"
df = pd.read_csv(file)

#To avoid showing the warning sign
st.set_option('deprecation.showPyplotGlobalUse', False)

#Big title
st.header("My first Streamlit App for Mall Dataset")

#For interactive purposes
option = st.sidebar.selectbox(
    'Select a mini project',
     ['Whole data', 'Description','KMeans cluster'])

#Shows the whole data
if option=='Whole data':
    whole = df
    st.table(whole)

#Shows the description table
elif option=='Description':
    chart_data = df.describe()
    st.table(chart_data)
    
#shows the kMeans Cluster plot-can start here without "else" if want to show static plot
else:
    st.write("KMeans Cluster Plot")
    #Preprocessing
    features = ['Annual_Income_(k$)', 'Spending_Score']
    X = df[features]
    plt.scatter(X['Annual_Income_(k$)'], X['Spending_Score']);
    kmeans = KMeans(n_clusters=5)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    plt.scatter(X['Annual_Income_(k$)'], X['Spending_Score'], c=y_kmeans, s=50, cmap='viridis')
    centers = kmeans.cluster_centers_
    plt.scatter(centers[:,0], centers[:,1], c='black', s=200, alpha=0.5);
    st.pyplot()
    
