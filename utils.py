import re
import math
import tensorflow as tf
import pandas as pd
import numpy as np
from scipy.spatial.distance import mahalanobis


# Replace NaN values with a default value appropriate for the data type
def fill_nan(data):
    for key, dtype in zip(data.keys(), data.dtypes):
        default = 'missing' if dtype == 'object' else 0
        data[key] = data[key].fillna(default)

        if dtype == 'object':
            data[key] = data[key].replace({'': default})
    return data


# Process data into Tensorflow Datasets and batch/cache data
def process_diamond_data(data, train_percent=0.8):
    data_len = data.shape[0]

    # Convert data into a Tensorflow Dataset. Later used to make train and test data
    data_ds = tf.data.Dataset.from_tensor_slices(dict(data))

    # Convert again, but to a constant Tensor, so it can be passed to the model to make a prediction after training.
    # Do this because there's some wackiness with how the Tensor shape is when converting it above.
    data_ds_to_predict = tf.data.Dataset.from_tensors({key: tf.constant(value) for key, value in data.items()})

    # Randomly shuffle the data.
    tf.random.set_seed(42)
    shuffled = data_ds.shuffle(data_len, seed=42, reshuffle_each_iteration=False)

    # Get number of data to train on and test on
    train_num = math.floor(data_len * train_percent)
    test_num = math.floor(data_len * (1 - train_percent))

    # Take number of data to train/test with shuffled data.
    train = data_ds.take(train_num)
    test = data_ds.skip(train_num).take(test_num)

    # Batch and cache data sets
    bc_train = train.batch(math.floor(train_num), drop_remainder=True).cache()
    bc_test = test.batch(math.floor(test_num / 10), drop_remainder=True).cache()

    return bc_train, bc_test, data_ds_to_predict


def preprocess_book_test_data():
    # Removing unnecessary features
    data = pd.read_csv('data/raw/books_sensitive.csv').drop(
        ['author_genres', 'author_average_rating', 'author_id', 'author_page_url', 'book_fullurl',
         'book_id', 'score', 'author_rating_count', 'author_review_count'], axis=1)

    # Convert publish date to int indicating year
    data['publish_date'].replace({'by': '0'}, inplace=True)
    replace_dict = {}
    for d in data['publish_date'].unique():
        if (type(d) != str and math.isnan(d)) or re.match(r'\d+st|\d+nd|\d+th|\d+rd', d):
            replace_dict[d] = 0
        else:
            replace_dict[d] = int(d.split(' ')[-1])

    data['publish_date'].replace(replace_dict, inplace=True)

    # Merge genre_1 and genre_2 into genre
    data['genre'] = [', '.join([genre_1, genre_2]) if genre_1 != genre_2 else genre_1
                     for genre_1, genre_2 in zip(data['genre_1'].values, data['genre_2'].values)]
    data.drop(['genre_1', 'genre_2'], axis=1, inplace=True)

    # Change type to appease Tensorflow
    data['pages'].replace({'1 page': 1}, inplace=True)
    data['pages'] = data['pages'].astype(float)

    # Rename features to match that of the training data
    data.rename(columns={
        'author_name': 'author',
        'book_title': 'title',
        'book_average_rating': 'average_rating',
        'num_ratings': 'rating_count',
        'num_reviews': 'review_count',
        'publish_date': 'date_published'
    }, inplace=True)

    # Remove whitespace from string values, and handle nans
    data['birthplace'] = data['birthplace'].str.strip()
    data['title'] = data['title'].str.strip()
    data = fill_nan(data)
    return data


def preprocess_and_merge_book_train_data():
    # Removing unnecessary features
    books = pd.read_csv('data/raw/books.csv').drop(
        ['id', 'link', 'cover_link', 'author_link', 'isbn13', 'original_title', 'asin', 'characters', 'settings',
         'amazon_redirect_link', 'worldcat_redirect_link', 'recommended_books', 'description', 'publisher',
         'series', 'five_star_ratings', 'four_star_ratings', 'three_star_ratings', 'two_star_ratings',
         'one_star_ratings', 'books_in_series', 'awards'],
        axis=1
    )
    ratings = pd.read_csv('data/raw/ratings_preprocessed.csv').drop(
        ['publisher', 'location', 'age', 'book_title', 'book_author', 'year_of_publication', 'img_s', 'img_m', 'img_l',
         'Summary', 'Category', 'city', 'state', 'country', 'Language'],
        axis=1
    )

    # Merge both data sets by isbn and then drop it since we don't need it
    merged = pd.merge(books, ratings, on='isbn')
    merged = merged.drop(['isbn'], axis=1)

    # Convert date published to just the year
    merged['date_published'].replace({d: 0 if type(d) != str and math.isnan(d)
                                     else int(d.split(' ')[-1])
                                     for d in merged['date_published'].unique()}, inplace=True)

    # No longer doing these.
    # # Count number of awards the book has and the number of books in the book's series
    # merged['awards'].replace({a: 0 if type(a) != str and math.isnan(a)
    #                          else len(a.split(',')) for a in merged['awards'].unique()}, inplace=True)
    # merged['books_in_series'].replace({a: 0 if type(a) != str and math.isnan(a)
    #                                    else len(a.split(','))
    #                                    for a in merged['books_in_series'].unique()}, inplace=True)
    #
    # # Fill empty values with 0 or an empty string. Also, sometimes there's a random 9???
    # merged['Language'].replace({'9': 'missing'}, inplace=True)

    merged = fill_nan(merged)

    # Remove numbers from genre_and_votes and rename it to genre
    genre_map = {}
    for raw_genre in merged['genre_and_votes'].unique():
        a = raw_genre.split(', ')[:2]
        top_genres = [genre.rsplit(' ', 1)[0] for genre in a]
        split_on_hyphen = []
        for g in top_genres:
            result = g.replace('-', ',').split(',')
            split_on_hyphen.extend(result)
        genre_map[raw_genre] = ', '.join(np.unique(split_on_hyphen))
    merged['genre_and_votes'].replace(genre_map, inplace=True)

    # Change type to appease Tensorflow
    merged['rating'] = merged['rating'].astype(float)

    # Rename columns for simplicity
    merged.rename(columns={'genre_and_votes': 'genre', 'number_of_pages': 'pages'}, inplace=True)
    return merged


def oh_encode_genres(train, test):
    all_genres = []
    all_genres.extend(train['genre'].values)
    all_genres.extend(test['genre'].values)
    genres = ', '.join(all_genres).replace('user', '').replace('-', ', ')
    split_genres = np.unique(genres.split(', '))

    # TODO currently do nothing with these, debating oh encoding authors or just dropping them
    authors = ', '.join(train['author'].values)
    split_authors = np.unique(authors.split(', '))

    train[split_genres] = 0
    test[split_genres] = 0
    train = train.reset_index().copy()
    test = test.reset_index().copy()

    for data in [train, test]:
        for i, title in enumerate(data['title'].unique()):
            rows_with_title = data.loc[data['title'] == title]
            split_g = rows_with_title['genre'].values[0].replace('user', '').replace('-', ', ').split(', ')
            new_col_val = np.where(data['title'] == title, 1, 0)
            data[split_g] = pd.DataFrame({g: new_col_val for g in split_g})

    train.drop(['genre'], axis=1, inplace=True)
    test.drop(['genre'], axis=1, inplace=True)
    train.to_csv('data/preprocessed/merged.csv', index=False)
    test.to_csv('data/preprocessed/books_sensitive.csv', index=False)


def preprocess_all_data():
    oh_encode_genres(
        preprocess_and_merge_book_train_data(),
        preprocess_book_test_data()
    )


def mahalanobis_distance():
    data = pd.read_csv('diamonds.csv').iloc[:, [0, 4, 5, 6, 7, 8, 9]]
    subset = data.head(1000)
    mean = np.mean(data)
    inv_cov_data = np.linalg.inv(np.cov(data.values.T))
    inv_cov_subset = np.linalg.inv(np.cov(subset.values.T))
    m = [[mahalanobis(row, subset.iloc[j], inv_cov_subset) for j in range(i + 1, subset.shape[0])]
         for i, row in subset.iterrows()]
    print(data.head())


# Used to one-hot encode string features in data.
def oh_encode(data):
    to_oh_encode = [feat for feat in data.select_dtypes(['object'])]  # One hot encode string values
    encodings = [pd.get_dummies(data[feat], prefix=feat) for feat in to_oh_encode]  # Get encodings from Pandas

    data = data.drop(to_oh_encode, 1)  # Drop the features that are now one hot encoded

    # Reintroduce the encoded values
    for e in encodings:
        data = data.join(e)
    return data
