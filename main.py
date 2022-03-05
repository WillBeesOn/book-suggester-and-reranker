import math
import time
import re
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from ciso8601 import parse_datetime
from utils import process_diamond_data, fill_nan, oh_encode, preprocess_all_data
from RankingModel import RankingModel

from typing import Dict, Tuple
import tensorflow_datasets as tfds
import tensorflow_ranking as tfr


def tf_recommend():
    # diamonds = pd.read_csv('diamonds.csv')
    # train, test, predict = process_diamond_data(diamonds)
    #
    # model = RankingModel(diamonds, 'price')
    # model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1))
    # f = model.fit(train, epochs=100)
    # e = model.evaluate(test, return_dict=True)
    # p = model.predict(predict)
    # print('a')

    # data = pd.read_csv('../data sets/good_reads_gendered.csv')
    #
    # # TODO determine if I want author genres, name, and title.... they add a lot of ohe features
    # data = data.drop(['author_id', 'author_page_url', 'book_fullurl', 'author_genres', 'book_id',
    #                   'author_name', 'book_title'], axis=1)
    #
    # replace_dict = {}
    # for d in data['publish_date'].unique():
    #     if type(d) != str and math.isnan(d):
    #         replace_dict[d] = 0
    #     else:
    #         replace_dict[d] = int(d.split(' ')[-1])
    #
    # data['publish_date'].replace(replace_dict, inplace=True)
    # data = oh_encode(data)
    #
    # train, test = train_test_split(data, test_size=0.2)
    # train_y = train.pop('score')
    # test_y = test.pop('score')
    #
    # # Learn multiple dense layers.
    # model = tf.keras.Sequential([
    #         tf.keras.layers.Dense(64, activation="relu", input_shape=(data.shape[1] - 1,)),
    #         # tf.keras.layers.Dense(64, activation="relu"),
    #         tf.keras.layers.Dense(1)  # Make rating predictions in the final layer.
    # ])
    # model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    #     loss=tf.keras.losses.MeanSquaredError(),
    #     metrics=[tf.keras.metrics.RootMeanSquaredError()]
    # )
    #
    # data.pop('score')
    # f = model.fit(train, train_y, epochs=50)
    # e = model.evaluate(test, test_y, return_dict=True)
    # p = model.predict(data)
    # print('a')

    # Ratings data.
    ratings = tfds.load('movielens/100k-ratings', split="train")
    # Features of all the available movies.
    movies = tfds.load('movielens/100k-movies', split="train")

    # Select the basic features.
    ratings = ratings.map(lambda x: {
        "movie_title": x["movie_title"],
        "user_id": x["user_id"],
        "user_rating": x["user_rating"]
    })

    movies = movies.map(lambda x: x["movie_title"])
    users = ratings.map(lambda x: x["user_id"])

    user_ids_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
    user_ids_vocabulary.adapt(users.batch(1000))

    movie_titles_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
    movie_titles_vocabulary.adapt(movies.batch(1000))

    key_func = lambda x: user_ids_vocabulary(x["user_id"])
    reduce_func = lambda key, dataset: dataset.batch(100)
    ds_train = ratings.group_by_window(key_func=key_func, reduce_func=reduce_func, window_size=100)

    for x in ds_train.take(1):
        for key, value in x.items():
            print(f"Shape of {key}: {value.shape}")
            print(f"Example values of {key}: {value[:5].numpy()}")
            print()

    def _features_and_labels(
            x: Dict[str, tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
        labels = x.pop("user_rating")
        return x, labels

    ds_train = ds_train.map(_features_and_labels)

    ds_train = ds_train.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=32))

    for x, label in ds_train.take(1):
        for key, value in x.items():
            print(f"Shape of {key}: {value.shape}")
            print(f"Example values of {key}: {value[:3, :3].numpy()}")
            print()
        print(f"Shape of label: {label.shape}")
        print(f"Example values of label: {label[:3, :3].numpy()}")

    class MovieLensRankingModel(tf.keras.Model):

        def __init__(self, user_vocab, movie_vocab):
            super().__init__()

            # Set up user and movie vocabulary and embedding.
            self.user_vocab = user_vocab
            self.movie_vocab = movie_vocab
            self.user_embed = tf.keras.layers.Embedding(user_vocab.vocabulary_size(), 64)
            self.movie_embed = tf.keras.layers.Embedding(movie_vocab.vocabulary_size(), 64)

        def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
            # Define how the ranking scores are computed:
            # Take the dot-product of the user embeddings with the movie embeddings.

            user_embeddings = self.user_embed(self.user_vocab(features["user_id"]))
            movie_embeddings = self.movie_embed(self.movie_vocab(features["movie_title"]))

            return tf.reduce_sum(user_embeddings * movie_embeddings, axis=2)

    # Create the ranking model, trained with a ranking loss and evaluated with
    # ranking metrics.
    model = MovieLensRankingModel(user_ids_vocabulary, movie_titles_vocabulary)
    optimizer = tf.keras.optimizers.Adagrad(0.5)
    loss = tfr.keras.losses.get(
        loss=tfr.keras.losses.RankingLossKey.SOFTMAX_LOSS, ragged=True)
    eval_metrics = [
        tfr.keras.metrics.get(key="ndcg", name="metric/ndcg", ragged=True),
        tfr.keras.metrics.get(key="mrr", name="metric/mrr", ragged=True)
    ]
    model.compile(optimizer=optimizer, loss=loss, metrics=eval_metrics)
    model.fit(ds_train, epochs=3)

    # Get movie title candidate list.
    for movie_titles in movies.batch(2000):
        break

    # Generate the input for user 42.
    inputs = {
        "user_id": tf.expand_dims(tf.repeat("42", repeats=movie_titles.shape[0]), axis=0),
        "movie_title": tf.expand_dims(movie_titles, axis=0)
    }

    # Get movie recommendations for user 42.
    scores = model(inputs)
    titles = tfr.utils.sort_by_scores(scores, [tf.expand_dims(movie_titles, axis=0)])[0]
    print(f"Top 5 recommendations for user 42: {titles[0, :5]}")


class BookRanker(tf.keras.Model):
    def __init__(self, df_data, features, key_name, label_name):
        super().__init__()
        self.__label_name = label_name
        self.__features = features

        # Convert Pandas Dataframe into a mapped Tensorflow Dataset, so we can access features like a dictionary
        feats_with_label = features.copy()
        feats_with_label.append(label_name)
        self.__mapped_feats = tf.data.Dataset.from_tensor_slices(dict(df_data)).map(lambda x: {
            feat: x[feat] for feat in feats_with_label
        })

        # TODO will I need to generate a vocabulary for the validation data too (data with birthplace and gender)? Yes.
        self.__unique_tensors = {}  # Create dict of tensors, 1 per feature containing the unique values of each feature
        self.__mapped_unique_tensors = {}  # Map unique values. Used for adapting vocab to a batch.
        self.__feat_tensors = {}  # Create a dict of tensors, 1 per feature containing all the data for that feature
        self.__feat_vocab = {}  # Create dict of feature vocabularies used for embeddings
        self.__feat_embed = {}  # Create dict of feature embeddings
        for feat in features:
            self.__unique_tensors[feat] = tf.data.Dataset.from_tensor_slices({feat: df_data[feat].unique()})
            self.__mapped_unique_tensors[feat] = self.__unique_tensors[feat].map(lambda x: x[feat])
            self.__feat_tensors[feat] = tf.data.Dataset.from_tensor_slices({feat: df_data[feat]})

            # TODO only do strings for now. Need to find out how to do with numeric features.
            if df_data[feat].dtype == 'object':
                self.__feat_vocab[feat] = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
                self.__feat_vocab[feat].adapt(self.__mapped_unique_tensors[feat].batch(1000))
                self.__feat_embed[feat] = tf.keras.layers.Embedding(self.__feat_vocab[feat].vocabulary_size(), 64)

        key_func = lambda x: self.__feat_vocab[key_name](x[key_name])
        reduce_func = lambda key, dataset: dataset.batch(100)
        train = self.__mapped_feats.group_by_window(key_func=key_func, reduce_func=reduce_func, window_size=100)

        def get_feat_and_label(d):
            labels = d.pop('rating')
            return d, labels

        train = train.map(get_feat_and_label)
        print(train.element_spec)
        print('---')
        [print(name) for name in train.element_spec[0]]
        self.train_data = train.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=32))

    def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
        embeddings = None
        for feat in self.__features:
            if feat == self.__label_name:
                continue
            if embeddings is None and features[feat].dtype == tf.string:
                embeddings = self.__feat_embed[feat](self.__feat_vocab[feat](features[feat]))
            elif features[feat].dtype == tf.string:
                embeddings = embeddings * self.__feat_embed[feat](self.__feat_vocab[feat](features[feat]))

        # TODO the problem really is multiplying and summing multiple tensors result in loss not changing.
        #  What other operations can I do to combine all features? Multiply all features and then with user?
        ue = self.__feat_embed['user_id'](self.__feat_vocab['user_id'](features['user_id']))
        te = self.__feat_embed['title'](self.__feat_vocab['title'](features['title']))
        ge = self.__feat_embed['genre'](self.__feat_vocab['author'](features['genre']))
        return tf.reduce_sum(ue * ge, axis=2)


def lens_test():
    class MovieLensRankingModel(tf.keras.Model):
        def __init__(self, user_vocab, movie_vocab):
            super().__init__()

            # Set up user and movie vocabulary and embedding.
            self.user_vocab = user_vocab
            self.movie_vocab = movie_vocab
            self.user_embed = tf.keras.layers.Embedding(user_vocab.vocabulary_size(),
                                                        64)
            self.movie_embed = tf.keras.layers.Embedding(movie_vocab.vocabulary_size(),
                                                         64)

        def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
            # Define how the ranking scores are computed:
            # Take the dot-product of the user embeddings with the movie embeddings.

            user_embeddings = self.user_embed(self.user_vocab(features["user_id"]))
            movie_embeddings = self.movie_embed(
                self.movie_vocab(features["title"]))

            return tf.reduce_sum(user_embeddings * movie_embeddings, axis=2)

    data = pd.read_csv('data/preprocessed/merged.csv')
    book_gendered = pd.read_csv('data/preprocessed/books_sensitive.csv')
    data['user_id'] = data['user_id'].astype(str)

    # Convert data into a Dataset
    ds = tf.data.Dataset.from_tensor_slices(dict(data))
    ds_titles = tf.data.Dataset.from_tensor_slices({
        'title': data['title'].unique()
    })

    # Select basic features
    feats = ds.map(lambda x: {
        'title': x['title'],
        'user_id': x['user_id'],
        'rating': x['rating']
    })

    titles = ds_titles.map(lambda x: x['title'])
    users = feats.map(lambda x: x['user_id'])

    title_vocab = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
    title_vocab.adapt(titles.batch(1000))

    user_vocab = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
    user_vocab.adapt(users.batch(1000))

    key_func = lambda x: user_vocab(x["user_id"])
    reduce_func = lambda key, dataset: dataset.batch(100)
    ds_train = feats.group_by_window(key_func=key_func, reduce_func=reduce_func, window_size=100)

    def get_feat_and_label(d):
        labels = d.pop('rating')
        return d, labels

    ds_train = ds_train.map(get_feat_and_label)
    ds_train = ds_train.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=32))

    for x, label in ds_train.take(1):
        for key, value in x.items():
            print(f"Shape of {key}: {value.shape}")
            print(f"Example values of {key}: {value[:3, :3].numpy()}")
            print()
        print(f"Shape of label: {label.shape}")
        print(f"Example values of label: {label[:3, :3].numpy()}")

    # Create the ranking model, trained with a ranking loss and evaluated with
    # ranking metrics.
    model = MovieLensRankingModel(user_vocab, title_vocab)
    optimizer = tf.keras.optimizers.Adagrad(0.5)
    loss = tfr.keras.losses.get(loss=tfr.keras.losses.RankingLossKey.SOFTMAX_LOSS, ragged=True)
    eval_metrics = [
        tfr.keras.metrics.get(key="ndcg", name="metric/ndcg", ragged=True),
        tfr.keras.metrics.get(key="mrr", name="metric/mrr", ragged=True)
    ]
    model.compile(optimizer=optimizer, loss=loss, metrics=eval_metrics)
    model.fit(ds_train, epochs=10)

    # Get movie title candidate list.
    for book_titles in titles.batch(2000):
        break

    # Generate the input for user 42.
    inputs = {
        "user_id": tf.expand_dims(tf.repeat("42", repeats=book_titles.shape[0]), axis=0),
        "title": tf.expand_dims(book_titles, axis=0)
    }

    # Get movie recommendations for user 42.
    scores = model(inputs)
    titles = tfr.utils.sort_by_scores(scores, [tf.expand_dims(book_titles, axis=0)])[0]
    print(f"Top 10 recommendations for user 42: {titles[0, :10]}")


def my_own_ranker_test():
    features = ['title', 'author', 'genre', 'average_rating', 'user_id',
                'rating_count', 'review_count', 'date_published', 'pages']
    data = pd.read_csv('data/preprocessed/merged.csv').sample(frac=1)
    book_gendered = pd.read_csv('data/preprocessed/books_sensitive.csv')
    data['user_id'] = data['user_id'].astype(str)

    model = BookRanker(data, features, 'user_id', 'rating')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.5),
        loss=tfr.keras.losses.get(loss=tfr.keras.losses.RankingLossKey.SOFTMAX_LOSS, ragged=True),
        metrics=[
            tfr.keras.metrics.get(key="ndcg", name="metric/ndcg", ragged=True),
            tfr.keras.metrics.get(key="mrr", name="metric/mrr", ragged=True)
        ]
    )
    model.fit(model.train_data, epochs=20)


if __name__ == '__main__':
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # preprocess_all_data()
    data = pd.read_csv('data/preprocessed/merged.csv').sample(frac=1)
    book_gendered = pd.read_csv('data/preprocessed/books_sensitive.csv')

    # Set up data in sensitive book data
    gendered_author = book_gendered.pop('author')
    gendered_birthplace = book_gendered.pop('birthplace')
    gendered_gender = book_gendered.pop('author_gender')
    gendered_title = book_gendered.pop('title')

    for user in data['user_id'].unique():
        user_data = data[data['user_id'] == user].copy()
        user_data.drop(['user_id'], axis=1, inplace=True)

        # Just using user_data as the training, no splitting for validation since there are so few ratings per person.
        # Just going to validate on the sensitive data.
        train_y = user_data.pop('rating')
        train_authors = user_data.pop('author')
        train_titles = user_data.pop('title')

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(user_data.shape[1], activation='relu', input_shape=(user_data.shape[1],)),
            tf.keras.layers.Dense(math.floor(user_data.shape[1] / 2)),
            tf.keras.layers.Dense(1)
        ])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=1000, min_delta=0.00001, baseline=4, restore_best_weights=True)
        ]
        model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.0001),
                      loss=tf.keras.losses.MeanAbsoluteError(),
                      # metrics=[tf.keras.metrics.Accuracy()],
                      )

        train_hist = model.fit(user_data, train_y, callbacks=callbacks, epochs=10000)
        h = pd.DataFrame(train_hist.history)
        h['epoch'] = train_hist.epoch

        # Get predicted values
        train_pred = model.predict(user_data)
        ranking_train = [(t, p[0]) for t, p in zip(train_titles, train_pred)]
        ranking_train.sort(key=lambda x: x[1])

        test_pred = model.predict(book_gendered)
        ranking_test = [(t, p[0]) for t, p in zip(gendered_title, test_pred)]
        ranking_test.sort(key=lambda x: x[1])
        print(ranking_train)
        print(ranking_test[:10])

    # print(train_hist.history['loss'])
    # plt.plot(train_hist.history['loss'], label='loss')
    # plt.ylim([0, 10])
    # plt.xlabel('Epoch')
    # plt.ylabel('Error [rating]')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # plt.scatter(train, train_y, label='Data')
    # linspace = tf.linspace(min(train), max(train), len(train))
    # # plt.plot(linspace, model.predict(linspace), color='k', label='Predictions')
    # plt.xlabel(var)
    # plt.ylabel('Rating')
    # plt.legend()
    # plt.show()
    #

