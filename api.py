from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import ast
import os

app = Flask(__name__)

def load_data():
    raw_data = pd.read_csv('tmdb_5000_movies.csv')
    cast_data = pd.read_csv('tmdb_5000_credits.csv')
    joined_data = raw_data.merge(cast_data,left_on='id', right_on='movie_id', how='inner')
    print(joined_data.head(1))
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
    final_data.to_csv('final_data.csv')
        
def split_data():
    data=pd.read_csv('final_data.csv')
    y=data['revenue'].values
    final_data_x=data.drop('revenue',axis=1)
    x=final_data_x.values
    return x,y

def train_data():
    x,y=split_data()
    
    
    
def linear_model():
        


@app.route('/linear', methods=['GET'])
def linear():
    model=linear_model()
    return {'msg':'work'}
    

if __name__ == '__main__':
    file_path = 'final_data.csv'
    if os.path.exists(file_path):
        print("Data already created")
    else:
        load_data()
    app.run(debug=True)
    