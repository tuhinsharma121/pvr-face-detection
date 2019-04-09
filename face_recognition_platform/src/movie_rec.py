import json
import numpy as np
import operator
import pandas as pd
import pickle
from flask import Flask, redirect, url_for, request
from operator import itemgetter

df = np.genfromtxt('models/Movie-Data.csv', delimiter=',')
df = df[1:, :]


def find_age_range(age):
    first = int(age / 10)
    if age < 18:
        return '13-17'
    elif age >= 18 and age < 25:
        return '18-24'
    elif age > 54 and age < 65:
        return '55-64'
    elif age >= 65:
        return '65+'
    if age % 10 >= 5:
        return str(first) + '5-' + str(first + 1) + '4'
    else:
        return str(first - 1) + '5-' + str(first) + '4'


def food_cold_no_login():
    sums = np.sum(df[:, 3:7], axis=0, )
    food_dict = {"samosa": sums[0], "popcorn-tub": sums[1], "singe-popcorn": sums[2], "coke": sums[3]}
    data = sorted(food_dict.items(), key=operator.itemgetter(1), reverse=True)
    final_data = []
    for key in data:
        final_data.append({"name": key[0], "frequency": key[1], "price": "", "url": ""})

    return json.dumps(final_data)


def movie_cold_no_login():
    data = pd.read_csv("models/Movie-Data.csv", index_col=0)
    sums = {}
    for column in data.columns[6:-1]:
        sums[str(column)] = int(data[column].value_counts()[1])
    sorted_s = sorted(sums.items(), key=operator.itemgetter(1), reverse=True)
    sorted_sums = []
    for i in sorted_s:
        sorted_sums.append({'name': i[0], 'frequency': i[1], 'url': ''})
    return json.dumps(sorted_sums)


def food_cold_first_login(user):
    data = pd.read_csv("models/Movie-Data.csv", index_col=0)
    data['age'] = data['age'].astype(str)
    age = data.iloc[user]['age']
    print(age)
    required = data[(data.age == age)]
    sums = {}
    for column in required.columns[2:6]:
        sums[str(column)] = int(required[column].value_counts()[1])
    sorted_s = sorted(sums.items(), key=operator.itemgetter(1), reverse=True)
    sorted_sums = []
    for i in sorted_s:
        sorted_sums.append({'name': i[0], 'frequency': i[1], 'url': ''})
    return json.dumps(sorted_sums)


def movie_first_login(user):
    data = pd.read_csv("models/Movie-Data.csv", index_col=0)
    data['age'] = data['age'].astype(str)
    data['gender'] = data['gender'].astype(str)
    Age = data.iloc[user]['age']
    gndr = data.iloc[user]['gender']
    print(Age, gndr)
    required = data[(data.age == Age) & (data.gender == gndr)][list(data.columns)]
    sums = {}
    for column in required.columns[6:-1]:
        sums[str(column)] = int(required[column].value_counts()[1])
    sorted_s = sorted(sums.items(), key=operator.itemgetter(1), reverse=True)
    sorted_sums = []
    for i in sorted_s:
        sorted_sums.append({'name': i[0], 'frequency': i[1], 'url': ''})
    return json.dumps(sorted_sums)


def food_personalised(user):
    df1 = pd.read_csv("models/Movie-Data.csv", index_col=0)
    df2 = df1.loc[df1['User_ID'] == user, 'samosa':'coke']
    d = {"samosa": df2.iloc[0]['samosa'], "popcorn_tub": df2.iloc[0]['popcorn_tub'], "popcorn_simple": df2.iloc[0]['popcorn_simple'],
         "coke": df2.iloc[0]['coke']}
    sorted_x = sorted(d.items(), key=operator.itemgetter(1), reverse=True)
    sorted_sums = []
    for i in sorted_x:
        sorted_sums.append({'name': i[0], 'frequency': i[1], 'url': ''})
    return json.dumps(sorted_sums, default=str)


def updating(user, item, movie, age, gender):
    df1 = pd.read_csv("models/Movie-Data.csv", index_col=0)
    df1['User_ID'] = df1.User_ID.astype(object)

    if len(df1.loc[df1['User_ID'] == user, 'samosa':'Irumbu Thirai']):
        x = df1.iloc[user][item]
        x = x + 1
        df1.ix[user, item] = x
        y = df1.iloc[user][movie]
        if y == 0:
            y = y + 1
            df1.ix[user, movie] = y
    df1.to_csv("models/Movie-Data.csv")


def movie_personalised(user):
    df = pd.read_csv('models/Movie-Data.csv', index_col=0)
    data_frame = pd.read_json('models/movie_data.json', orient='records')  # for list
    df['User_ID'] = df.User_ID.astype(object)
    movies_count = df.loc[df['User_ID'] == user, 'Avengers: Infinity War':'Irumbu Thirai']
    df2 = list(movies_count.columns)
    dict1 = {df.iloc[0][df2[0]]: df2[0]}  # watched_or_not,Movie name

    for i in range(1, 26):
        dict1.update({df2[i]: df.iloc[0][df2[i]]})

    # To get genres in each movie
    movie_genres = []
    all_genre = []
    for i in range(0, 26):
        movie_genres.append(data_frame.iloc[i][3])

    # To get all kind of genres in entire movie set
    for i in range(0, 26):
        length = len(movie_genres[i])
        for j in range(0, length):
            if movie_genres[i][j] not in all_genre:
                all_genre.append(movie_genres[i][j])

    len_genres = len(all_genre)

    user_genre = []
    movie_name = []
    count = 0
    # To get all movie_names list
    for key, value in dict1.items():
        if (value == 1):
            movie_name.append(key)
            count += 1

    # To get genre of the movies that the user has watched
    for i in range(0, 26):
        movie = data_frame.iloc[i][6]
        for j in range(0, count):
            if movie_name[j] == movie:
                user_genre.append(data_frame.iloc[i][3])

    freq = {}

    # Count the frequency of each genre for the user
    for i in range(0, count - 1):
        length_user_genre = len(user_genre[i])
        for j in range(0, length_user_genre):
            if user_genre[i][j] not in freq.keys():
                freq[user_genre[i][j]] = 1
            else:
                freq[user_genre[i][j]] = freq[user_genre[i][j]] + 1

    list_movies = {}

    for i in range(6, 32):
        if df.iloc[user][i] != 1:
            length = len(data_frame.iloc[i - 6][3])
            for j in range(0, length):
                if data_frame.iloc[i - 6][3][j] in freq.keys() and movies_count.columns[i - 6] not in list_movies.keys():
                    list_movies[i - 6] = df2[i - 6]

    listt = list(list_movies.values())
    review_rate = []

    index = list(list_movies.keys())

    for i in list_movies.keys():
        if data_frame.iloc[i][9] == "null":
            review_rate.append(0.0)
        else:
            review_rate.append(float(data_frame.iloc[i][9]))
    to_dump = []
    for i in range(0, len(index)):
        to_dump.append({"name": listt[i], "review_rate": (review_rate[i])})

    newlist = sorted(to_dump, key=lambda k: k['review_rate'], reverse=True)

    return json.dumps(newlist)
