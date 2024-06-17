from flask import Flask, render_template, request
import pandas as pd
import requests

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('movie.html')

@app.route('/about')
def movie():
    return render_template('about.html')

@app.route('/faq')
def faq():
    return render_template('faq.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    movie_title = request.form['Title']
    movie_genre = request.form.getlist('movieGenre')
    country = request.form.getlist('country')
    release_date = request.form['releaseDate']
    budget = request.form['budget']
    vote_avg = request.form['Voteavg']
    vote_num = request.form['Votenum']
    runtime = request.form['runtime']
    release_day = request.form.getlist('releaseDay')

    # For demonstration, we'll just print the data to console
    print(f"Title: {movie_title}")
    print(f"Genre: {movie_genre}")
    print(f"Country: {country}")
    print(f"Release Date: {release_date}")
    print(f"Budget: {budget}")
    print(f"Release Day: {release_day}")
    input_data = {
        "title_x": movie_title,
        "budget": budget,
        "vote_average":vote_avg,
        "vote_count":vote_num,
        "runtime":runtime,
        "genres": movie_genre,
        "production_countries": country,
        "release_date": release_date,
        'star_power':43,
        "release_day": release_day
    }

    try:
        response = requests.post('http://localhost:5001/linear', json=input_data)
        response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
        revenue = response.json()
        print('Predicted Revenue:', revenue)
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

    
    # Normally, you'd use this data to make a prediction or store it in a database
    # Here, we just pass it to a new page to display
    return render_template('result.html', title=movie_title, genre=movie_genre,
                           country=country, release_date=release_date,
                           budget=budget, release_day=release_day,revenue=100000)

if __name__ == '__main__':
  
    app.run(debug=True)
