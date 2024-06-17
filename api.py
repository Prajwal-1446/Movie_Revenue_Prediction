from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import ast
import os
from sklearn.model_selection import train_test_split as tts
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,accuracy_score
from sklearn.linear_model import LinearRegression
from  sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA

app = Flask(__name__)

def load_data():
    raw_data = pd.read_csv('tmdb_5000_movies.csv')
    cast_data = pd.read_csv('tmdb_5000_credits.csv')
    joined_data = raw_data.merge(cast_data,left_on='id', right_on='movie_id', how='inner')
    data=joined_data[['title_x','budget','revenue','vote_average','vote_count','runtime','genres','spoken_languages','production_countries','release_date','cast']].copy()
    data.dropna(inplace=True)
    data = data[data['revenue']>=100000]
    df = pd.DataFrame(data)
    
    def extract_genre_names(genres_str):
        genres_list = ast.literal_eval(genres_str)
        return [genre['name'] for genre in genres_list]
    df['genres'] = df['genres'].apply(extract_genre_names)
    df_exploded = df.explode('genres')
    df_one_hot = pd.get_dummies(df_exploded, columns=['genres'], prefix='', prefix_sep='')
    df_one_hot = df_one_hot.groupby('title_x').max().reset_index()
    df_one_hot.set_index('title_x', inplace=True)
    
    df = pd.DataFrame(df_one_hot)
    def extract_names(json_str):
        items_list = ast.literal_eval(json_str)
        return [item['name'] for item in items_list]
    df['spoken_languages'] = df['spoken_languages'].apply(extract_names)
    df_exploded_languages = df.explode('spoken_languages')
    df_one_hot_languages = pd.get_dummies(df_exploded_languages, columns=['spoken_languages'], prefix='', prefix_sep='')
    df_one_hot_languages = df_one_hot_languages.groupby('title_x').max().reset_index()
    df_one_hot_languages.set_index('title_x',inplace=True)
    
    df = pd.DataFrame(df_exploded_languages)
    def extract_country_names(country_str):
        country_list = ast.literal_eval(country_str)
        return [country['name'] for country in country_list]
    df['production_countries'] = df['production_countries'].apply(extract_country_names)
    df_exploded = df.explode('production_countries')
    df_one_hot = pd.get_dummies(df_exploded, columns=['production_countries'], prefix='', prefix_sep='')
    df_one_hot = df_one_hot.groupby('title_x').max().reset_index()
    df_one_hot.set_index('title_x', inplace=True)
    
    df_combined=df_one_hot.copy()
    df_combined['cast']
    df = pd.DataFrame(df_combined)
    def count_actor_occurrences(data):
        actor_counts = {}
        for row in data:
            cast_list = ast.literal_eval(row)
            for actor in cast_list:
                actor_name = actor['name']
                actor_counts[actor_name] = actor_counts.get(actor_name, 0) + 1
        return actor_counts
    actor_counts = count_actor_occurrences(df['cast'])
    def calculate_star_power(row):
        cast_list = ast.literal_eval(row)
        leading_cast = cast_list[:2]  # Consider only the first two cast members
        star_power = sum(actor_counts.get(actor['name'], 0) for actor in leading_cast)
        return star_power
    df['star_power'] = df['cast'].apply(calculate_star_power)
    df.drop(columns=['cast'], inplace=True)
    df.drop('spoken_languages', axis=1, inplace=True)
    
    data=df.copy()
    df = pd.DataFrame(data)
    df['release_date'] = pd.to_datetime(df['release_date'])
    df['Day of Week'] = df['release_date'].dt.day_name()
    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for day in days_of_week:
        df[day] = (df['Day of Week'] == day).astype(int)
    df.drop(columns=['release_date', 'Day of Week'], inplace=True)
    pd.set_option('display.max_columns', None)
    final_data=df.copy()
    print(final_data.keys)
    final_data.to_csv('final_data.csv',index='title_x')
        
def split_data():
    data=pd.read_csv('final_data.csv')
    data.drop(columns='title_x',inplace=True)
    print(data.keys)
    y=data['revenue'].values
    final_data_x=data.drop('revenue',axis=1)
    x=final_data_x.values
    return x,y

def train_data():
    print("train")
    x,y=split_data()
    xtrain,xtest,ytrain,ytest=tts(x,y,test_size=1/3)
    return xtrain,xtest,ytrain,ytest
    
def train_split(x,y):
    xtrain,xtest,ytrain,ytest=tts(x,y,test_size=1/3)
    return xtrain,xtest,ytrain,ytest
            
    
def linear_model():
    xtrain,xtest,ytrain,ytest=train_data()
    model = LinearRegression()
    model.fit(xtrain,ytrain)
    ypred=model.predict(xtest)
    print("MAE",mean_absolute_error(ytest,ypred))
    print("MSE",mean_squared_error(ytest,ypred))
    print("RMSE",np.sqrt(mean_squared_error(ytest,ypred)))
    print("R2_Score",r2_score(ytest,ypred))
    return model

    
def poly_mod():
    x,y=split_data()
    pca = PCA(n_components=50)
    x_pca = pca.fit_transform(x)
    poly = PolynomialFeatures(degree = 2, include_bias=True)
    x_poly=poly.fit_transform(x_pca)
    xtrain,xtest,ytrain,ytest=train_split(x_poly,y)
    polymodel=LinearRegression()
    polymodel.fit(xtrain,ytrain)
    ypred=polymodel.predict(xtest)
    print("MAE",mean_absolute_error(ytest,ypred))
    print("MSE",mean_squared_error(ytest,ypred))
    print("RMSE",np.sqrt(mean_squared_error(ytest,ypred)))
    print("R2_Score",r2_score(ytest,ypred))
    return polymodel


@app.route('/linear', methods=['POST'])
def linear():
    model = linear_model()  # Assuming this function initializes and trains your model
    input_data = request.json  # Assuming JSON data is sent in the request
    input_df = pd.DataFrame([input_data])
    
    # Initialize all genre, country, and day columns to False
    genres = [
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Family",
    "Fantasy",
    "Foreign",
    "History",
    "Horror",
    "Music",
    "Mystery",
    "Romance",
    "Science Fiction",
    "Thriller",
    "War","Western"]

    
    countries = [
    "Argentina",
    "Australia",
    "Austria",
    "Bahamas",
    "Belgium",
    "Bolivia",
    "Brazil",
    "Bulgaria",
    "Canada",
    "Chile",
    "China",
    "Czech Republic",
    "Denmark",
    "Dominica",
    "Fiji",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hong Kong",
    "Hungary",
    "Iceland",
    "India",
    "Indonesia",
    "Iran",
    "Ireland",
    "Israel",
    "Italy",
    "Jamaica",
    "Japan",
    "Kazakhstan",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Mexico",
    "Monaco",
    "Morocco",
    "Netherlands",
    "New Zealand",
    "Norway",
    "Pakistan",
    "Peru",
    "Philippines",
    "Poland",
    "Portugal",
    "Romania",
    "Russia",
    "Serbia",
    "Serbia and Montenegro",
    "Singapore",
    "Slovenia",
    "South Africa",
    "South Korea",
    "Spain",
    "Sweden",
    "Switzerland",
    "Taiwan",
    "Thailand",
    "Tunisia",
    "United Arab Emirates",
    "United Kingdom",
    "United States of America"
]

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    
    for genre in genres:
        input_df[genre] = False
    for country in countries:
        input_df[country] = False
    for day in days:
        input_df[day] = False

    # Set the appropriate columns to True based on the input data
    for genre in input_data.get('genres', []):
        if genre in genres:
            input_df[genre] = True
    for country in input_data.get('production_countries', []):
        if country in countries:
            input_df[country] = True
    for day in input_data.get('release_day', []):
        if day in days:
            input_df[day] = True
    
    # Ensure the input_df has the same columns as the training data
    required_columns = [col for col in input_df.columns if col != 'revenue']  # Adjust based on your actual columns
    input_df = input_df[required_columns]
    columns_to_drop = ['title_x', 'genres', 'production_countries', 'release_date', 'release_day']
    input_df = input_df.drop(columns=[col for col in columns_to_drop if col in input_df.columns])
    # # Make prediction
    print(input_df.keys())

    prediction = model.predict(input_df.values)
    
    return jsonify({'prediction': prediction.tolist()})


@app.route('/poly', methods = ['POST'])
def polyy():
    model=poly_mod()
    input_data = request.json
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df.values)
    
    return jsonify({'prediction': prediction[0]})
@app.route('/ping', methods=['GET'])
def ping():
    return "Pong", 200
    
if __name__ == '__main__':
    file_path = 'final_data.csv'
    if os.path.exists(file_path):
        print("Data already created")
    else:
        load_data()
    app.run(host='127.0.0.1',port=5001,debug=True)
    