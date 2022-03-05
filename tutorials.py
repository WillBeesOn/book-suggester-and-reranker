import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_ranking as tfr
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import matplotlib.pyplot as plt
from tensorflow import keras
from typing import Dict, Text, Tuple
from factor_analyzer.factor_analyzer import FactorAnalyzer, calculate_kmo, calculate_bartlett_sphericity


def tensorflow():
    mnist = keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test)


def example_recommend():
    ratings = tfds.load("movielens/100k-ratings", split="train")

    ratings = ratings.map(lambda x: {
        "movie_title": x["movie_title"],
        "user_id": x["user_id"],
        "user_rating": x["user_rating"]
    })

    tf.random.set_seed(42)
    shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

    train = shuffled.take(80_000)
    test = shuffled.skip(80_000).take(20_000)

    movie_titles = ratings.batch(1_000_000).map(lambda x: x["movie_title"])
    user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

    cached_train = train.shuffle(100_000).batch(8192).cache()
    cached_test = test.batch(4096).cache()

    unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
    unique_user_ids = np.unique(np.concatenate(list(user_ids)))

    class RankingModel(tf.keras.Model):
        def __init__(self):
            super().__init__()
            embedding_dimension = 32

            # Compute embeddings for users.
            self.user_embeddings = tf.keras.Sequential([
                tf.keras.layers.StringLookup(
                    vocabulary=unique_user_ids, mask_token=None),
                tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
            ])

            # Compute embeddings for movies.
            self.movie_embeddings = tf.keras.Sequential([
                tf.keras.layers.StringLookup(
                    vocabulary=unique_movie_titles, mask_token=None),
                tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
            ])

            # Compute predictions.
            self.ratings = tf.keras.Sequential([
                # Learn multiple dense layers.
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(64, activation="relu"),
                # Make rating predictions in the final layer.
                tf.keras.layers.Dense(1)
            ])

        def call(self, inputs):
            user_id, movie_title = inputs

            user_embedding = self.user_embeddings(user_id)
            movie_embedding = self.movie_embeddings(movie_title)

            return self.ratings(tf.concat([user_embedding, movie_embedding], axis=1))

    class MovielensModel(tfrs.models.Model):
        def __init__(self):
            super().__init__()
            self.ranking_model: tf.keras.Model = RankingModel()
            self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
                loss=tf.keras.losses.MeanSquaredError(),
                metrics=[tf.keras.metrics.RootMeanSquaredError()]
            )

        def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
            return self.ranking_model((features["user_id"], features["movie_title"]))

        def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
            labels = features.pop("user_rating")

            rating_predictions = self(features)

            # The task computes the loss and the metrics.
            return self.task(labels=labels, predictions=rating_predictions)

    model = MovielensModel()
    model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

    f = model.fit(cached_train, epochs=2)
    e = model.evaluate(cached_test, return_dict=True)
    p = model.predict(cached_train)

    test_ratings = {}
    test_movie_titles = ["M*A*S*H (1970)", "Dances with Wolves (1990)", "Speed (1994)"]
    for movie_title in test_movie_titles:
        test_ratings[movie_title] = model({
            "user_id": np.array(["42"]),
            "movie_title": np.array([movie_title])
        })

    print("Ratings:")
    for title, score in sorted(test_ratings.items(), key=lambda x: x[1], reverse=True):
        print(f"{title}: {score}")


def ranking_tutorial_2():
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
                self.movie_vocab(features["movie_title"]))

            return tf.reduce_sum(user_embeddings * movie_embeddings, axis=2)


    class DiamondRankingModel(tf.keras.Model):
        def __init__(self, cut_v, color_v, clarity_v, key_v):
            super().__init__()

            # Set up user and movie vocabulary and embedding.
            self.cut = cut_v
            self.cut_e = tf.keras.layers.Embedding(cut_v.vocabulary_size(), 64)
            self.color = color_v
            self.color_e = tf.keras.layers.Embedding(color_v.vocabulary_size(), 64)
            self.clarity = clarity_v
            self.clarity_e = tf.keras.layers.Embedding(clarity_v.vocabulary_size(), 64)
            self.key = key_v
            self.key_e = tf.keras.layers.Embedding(key_v.vocabulary_size(), 64)

        def call(self, features: Dict[str, tf.Tensor]) -> tf.Tensor:
            # Define how the ranking scores are computed:
            # Take the dot-product of the user embeddings with the movie embeddings.

            key_embeddings = self.key_e(self.key(features["key"]))
            color_embeddings = self.color_e(self.color(features["color"]))
            clarity_embeddings = self.clarity_e(self.clarity(features["clarity"]))

            return tf.reduce_sum(key_embeddings * clarity_embeddings * color_embeddings, axis=2)

    # Ratings data.
    ratings = tfds.load('movielens/100k-ratings', split="train")
    # Features of all the available movies.
    movies = tfds.load('movielens/100k-movies', split="train")

    # Load diamonds and turn it into dataset
    diamonds = pd.read_csv('diamonds.csv')
    diamonds['key'] = [f'{i}' for i in range(diamonds.shape[0])]
    diamonds['price'] = [float(v) for v in diamonds['price'].values]  # Convert prices to float to prevent cast error
    d_ds = tf.data.Dataset.from_tensor_slices(dict(diamonds))

    # ==================================

    # Select the basic features.
    ratings = ratings.map(lambda x: {
        "movie_title": x["movie_title"],
        "user_id": x["user_id"],
        "user_rating": x["user_rating"]
    })

    # Select diamond features
    d_feats = d_ds.map(lambda x: {
        # "cut": x["cut"],
        "color": x["color"],
        "clarity": x["clarity"],
        "price": x["price"],
        "key": x["key"]
    })

    # ==================================

    # Get just movies and user_ids
    movies = movies.map(lambda x: x["movie_title"])
    users = ratings.map(lambda x: x["user_id"])

    # Map feats to use for vocab embeddings
    cut = d_ds.map(lambda x: x["cut"])
    color = d_ds.map(lambda x: x["color"])
    clarity = d_ds.map(lambda x: x["clarity"])
    key = d_ds.map(lambda x: x["key"])

    # ==================================

    # Don't need price vocab since that's just used for computing loss
    cut_vocab = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
    cut_vocab.adapt(cut.batch(1000))

    color_vocab = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
    color_vocab.adapt(color.batch(1000))

    clarity_vocab = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
    clarity_vocab.adapt(clarity.batch(1000))

    key_vocab = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
    key_vocab.adapt(key.batch(1000))

    # Create embedding for user ids
    user_ids_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
    user_ids_vocabulary.adapt(users.batch(1000))

    # Create embedding for movie titles
    movie_titles_vocabulary = tf.keras.layers.experimental.preprocessing.StringLookup(mask_token=None)
    movie_titles_vocabulary.adapt(movies.batch(1000))

    # ==================================

    # User ids are used for keys to group data for each observation
    key_func = lambda x: user_ids_vocabulary(x["user_id"])

    # Reduce data set to a batch of 100
    reduce_func = lambda key, dataset: dataset.batch(100)

    # Create individual observations by grouping them by user_id, and create batches of 100.
    ds_train = ratings.group_by_window(key_func=key_func, reduce_func=reduce_func, window_size=100)

    # Probably can use the same reduce func
    dkey_func = lambda x: key_vocab(x["key"])
    diamond_train = d_feats.group_by_window(key_func=dkey_func, reduce_func=reduce_func, window_size=100)

    # ==================================

    def _features_and_labels(x: Dict[str, tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
        labels = x.pop("user_rating")
        return x, labels

    ds_train = ds_train.map(_features_and_labels)
    ds_train = ds_train.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=32))

    def _d_features_and_labels(x: Dict[str, tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], tf.Tensor]:
        labels = x.pop("price")
        return x, labels

    diamond_train = diamond_train.map(_d_features_and_labels)
    diamond_train = diamond_train.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=32))

    # ==================================

    # Create the ranking model, trained with a ranking loss and evaluated with
    # ranking metrics.
    # model = MovieLensRankingModel(user_ids_vocabulary, movie_titles_vocabulary)
    model = DiamondRankingModel(cut_vocab, color_vocab, clarity_vocab, key_vocab)
    optimizer = tf.keras.optimizers.Adagrad(0.5)
    loss = tfr.keras.losses.get(loss=tfr.keras.losses.RankingLossKey.SOFTMAX_LOSS, ragged=True)
    eval_metrics = [
        tfr.keras.metrics.get(key="ndcg", name="metric/ndcg", ragged=True),
        tfr.keras.metrics.get(key="mrr", name="metric/mrr", ragged=True)
    ]
    model.compile(optimizer=optimizer, loss=loss, metrics=eval_metrics)
    f = model.fit(diamond_train, epochs=3)
    # f = model.fit(ds_train, epochs=3)

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


# Factor Analysis Tutorial
def factor_analysis():
    df = pd.read_csv('bfi.csv')
    df.drop(['gender', 'education', 'age'], axis=1, inplace=True)
    df.dropna(inplace=True)

    print(calculate_bartlett_sphericity(df))
    print(calculate_kmo(df))

    fa = FactorAnalyzer(25, rotation=None)
    print(fa.fit(df))
    ev, v = fa.get_eigenvalues()
    print(ev)

    plt.scatter(range(1, df.shape[1] + 1), ev)
    plt.plot(range(1, df.shape[1] + 1), ev)
    plt.title('Scree Plot')
    plt.xlabel('Factors')
    plt.ylabel('Eigenvalue')
    plt.grid()
    plt.show()

    print('------')
    fa2 = FactorAnalyzer(5, rotation='varimax')
    fa2.fit(df)
    print(fa2.loadings_)
    print(fa2.get_factor_variance())


