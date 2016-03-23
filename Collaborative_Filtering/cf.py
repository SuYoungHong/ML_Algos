import csv
import numpy as np
import argparse

def file_reader(filepath):
    """
    :arg
    path to csv file, csv file will be of form:
        user,movie,rating
    :returns
    (csvdict, userset, movieset) in a tuple
        csvdict: is a dictionary of dict[user][movie] = rating
        userset: is a set of all users
        movieset: is a set of all movies
    """
    csvdict = {}
    userset = set()
    movieset = set()
    with open(filepath, 'r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            if rows[0] not in userset:
                csvdict[rows[0]] = {rows[1]:float(rows[2])}
                userset.add(rows[0])
            else:
                csvdict[rows[0]][rows[1]] = float(rows[2])
            movieset.add(rows[1])
    return (csvdict, userset, movieset)

def get_averages(filepath):
    """
    :arg
    path to csv file
    file should be in the form:
        user,movie,rating
    :returns
    (useravg, movieavg, totalavg) in a tuple
        useravg: is a dictionary of dict[user] = average overall rating
        movieavg: is a dictionary of dict[movie] = average overall rating
        totalavg: is a float, for overall rating of user-movie ratings
    """
    userratings = {}
    movieratings = {}
    totalratings = []
    userset = set()
    movieset = set()
    with open(filepath, 'r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            if rows[0] not in userset:
                userratings[rows[0]] = [float(rows[2])]
                userset.add(rows[0])
            else:
                userratings[rows[0]].append(float(rows[2]))
            if rows[1] not in movieset:
                movieratings[rows[1]] = [float(rows[2])]
                movieset.add(rows[1])
            else:
                movieratings[rows[1]].append(float(rows[2]))
            totalratings.append(float(rows[2]))
    useravg = {}
    for user in userset:
        useravg[user] = np.mean(userratings[user])
    movieavg = {}
    for movie in movieset:
        movieavg[movie] = np.mean(movieratings[movie])
    totalavg = np.mean(totalratings)
    return (useravg, movieavg, totalavg)

def user_seen_sets(filepath):
    """
    :arg
    path to csv file
    file should be in the form:
        user,movie,rating
    :returns
    usr_mv_sets in a dictionary
        usr_mv_set: is dict[user] = set of movies seen for that user
    """
    usr_mv_sets = {}
    userset = set()
    with open(filepath, 'r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            if rows[0] not in userset:
                usr_mv_sets[rows[0]] = set()
                usr_mv_sets[rows[0]].add(rows[1])
                userset.add(rows[0])
            else:
                usr_mv_sets[rows[0]].add(rows[1])
    return usr_mv_sets

def films_user_sets(filepath):
    """
    :arg
    path to csv file
    file should be in the form:
        user,movie,rating
    :returns
    movie_to_user in a dictionary
        movie_to_user: is dict[movie] = set of users that have seen a movie per movie
    """
    movie_to_user = {}
    movieset = set()
    with open(filepath, 'r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            if rows[1] not in movieset:
                movie_to_user[rows[1]] = set()
                movie_to_user[rows[1]].add(rows[0])
                movieset.add(rows[1])
            else:
                movie_to_user[rows[1]].add(rows[0])
    return movie_to_user

def get_distance(user1, user2, usr_sets, usr_avgs, fulldict):
    """
    :arg
    user1, user2, usr_sets, usr_avgs, fulldict
        user1: is string
        user2: is string
        usr_sets: is dictionary of dict[user] = seen movie set
        usr_avgs: is dictionary of dict[user] = average movie rating
        fulldict: is dictionary of dict[user][movie] = rating per movie for each user
    :returns
    distance
        distance is an float describing pearson coefficient between the two users based on ratings of movies seen in
        common
    """
    commonset = usr_sets[user1] & usr_sets[user2]
    if commonset:
        topsum = 0.0
        bottomsum1 = 0.0
        bottomsum2 = 0.0
        for movie in commonset:
            user1_diff = fulldict[user1][movie] - usr_avgs[user1]
            user2_diff = fulldict[user2][movie] - usr_avgs[user2]
            topsum += (user1_diff * user2_diff)
            bottomsum1 += (user1_diff**2)
            bottomsum2 += (user2_diff**2)
        bottom = np.sqrt(bottomsum1 * bottomsum2)
        if bottom != 0.0:
            distance = topsum/bottom
        else:
            distance = 0.0
    else:
        distance = 0.0
    return distance

def train_data(filepath):
    """
    :arg
    path to csv file, should be in the form:
        user, movie, rating
    :returns
    (model, userset, movieset)
        model: is a dict of dicts of form dict[user] = {other user:similarity}
        userset: set of all users in training set
        movieset: set of all movies in training set
    """
    model = {}
    fulldict, userset, movieset = file_reader(filepath)
    useravg, movieavg, totalavg = get_averages(filepath)
    userseen_sets = user_seen_sets(filepath)
    for user in userset:
        model[user] = {}
        compareset = set(userset)
        compareset.remove(user)
        for otheruser in compareset:
            model[user][otheruser] = get_distance(user, otheruser, userseen_sets, useravg, fulldict)
    return (model, userset, movieset)

def predict(user, movie, model, fulldata, useravg, film_user_set):
    """
    :param
    user, movie, model, fullset,useravg, film_user_set:
        user: string of user
        movie: string of movie
        model: dict of dict of pearson coeff per user between every other user
        fulldata: the original training data in dict[user][film]=rating
        useravg: dictionary of average ratings per user
        film_user_set: dictionary of dict[film]={set of all users who saw this film}
    :return
    (prediction, user, movie)
        prediction: rating prediction, a string
        user: user
        movie: movie
    """
    compare_set = film_user_set[movie]
    if user in compare_set:
        compare_set.remove(user)
    rightsum = 0
    coeff = 0
    for compare_user in compare_set:
        if model[user][compare_user]:
            jdiff = fulldata[compare_user][movie] - useravg[compare_user]
            rightsum += model[user][compare_user] * jdiff
            coeff += abs(model[user][compare_user])
    if coeff != 0:
        right = + (rightsum/coeff)
    else:
        right = 0
    prediction = useravg[user] + right
    return (prediction, user, movie)

def predict_list(filepath):
    """
    :arg
    filepath
    :returns
    pred_list
        pred_list: list of user, movie pairs in form [(user, movie), (user,movie), etc.]
    """
    pred_list = []
    with open(filepath, 'r') as infile:
        reader = csv.reader(infile)
        for rows in reader:
            pred_list.append((rows[0], rows[1], rows[2]))
    return pred_list

def check_membership(user, movie, userset, movieset):
    """
    :param user: string
    :param movie: string
    :param userset: set of all users (training)
    :param movieset: set of all movies (training)
    :return: int: 1 means has both, 2 means has just user, 3 means has just movie, 4 means has neither
    """
    if user in userset and movie in movieset:
        result = 1 #both
    elif user in userset and movie not in movieset:
        result = 2 #just user
    elif user not in userset and movie in movieset:
        result = 3 #just movie
    else:
        result = 4 # neither
    return result

def test_data(model, filepath, filepath_testing):
    fulldata, userset, movieset = file_reader(filepath)
    useravg, movieavg, totalavg = get_averages(filepath)
    film_user_set = films_user_sets(filepath)
    result_list = []
    pred_list = predict_list(filepath_testing)
    for row in pred_list:
        row_id = (row[0],row[1], float(row[2]))
        row_type = check_membership(row[0], row[1], userset, movieset)
        if row_type == 1:
            finalrow = row_id + (predict(row[0], row[1], model, fulldata, useravg, film_user_set)[0],)
        elif row_type == 2:
            finalrow = row_id + (useravg[row[0]],)
        elif row_type == 3:
            finalrow = row_id + (movieavg[row[1]],)
        elif row_type == 4:
            finalrow = row_id + (totalavg,)
        result_list.append(finalrow)
    return result_list

def find_MSRE(prediction_list):
    total = 0.0
    for row in prediction_list:
        total += (row[3] - row[2])**2
    MSE = total/len(prediction_list)
    MSRE = np.sqrt(MSE)
    return MSRE

def find_MAE(prediction_list):
    total = 0.0
    for row in prediction_list:
        total += abs(row[3] - row[2])
    MAE = total/len(prediction_list)
    return MAE

def makefile(prediction_list):
    with open('predictions.txt', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(prediction_list)


if __name__== '__main__':
    parser = argparse.ArgumentParser(description='collaborate filtering algorithm')
    parser.add_argument('-t','--train', help='your training data csv', required=True)
    parser.add_argument('-T','--test', help='your testing data csv', required=True)
    args = vars(parser.parse_args())
    trainset =  args['train']
    testset =  args['test']
    model = train_data(trainset)[0]
    predicted = test_data(model, trainset, testset)
    makefile(predicted)
    print 'Mean Absolute Error is: ' + str(find_MAE(predicted))
    print 'Root Mean Squared Error is: ' + str(find_MSRE(predicted))
