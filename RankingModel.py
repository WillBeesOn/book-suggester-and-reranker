import tensorflow as tf
import tensorflow_recommenders as tfrs


class RankingModel(tfrs.models.Model):
    def __init__(self, data, label_name):
        super().__init__()

        self.__label_name = label_name  # Set the name of the feature which is the label

        # Compute embeddings for all features.
        embedding_dimension = 64
        # for key in data.keys():
        #     print(f'{key}: {len(data[key].unique())}')
        self.__embeddings = {
            feat: tf.keras.Sequential([
                tf.keras.layers.StringLookup(vocabulary=data[feat].unique(), mask_token=None),
                tf.keras.layers.Embedding(len(data[feat].unique()) + 1, embedding_dimension)
            ]) if dtype == 'object'
            else tf.keras.layers.Embedding(len(data[feat].unique()) + 1, embedding_dimension)
            for feat, dtype in zip(data.keys(), data.dtypes)}  # Embed/encode categorical string features

        # Create model to use.
        self.__model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),  # Learn multiple dense layers.
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dense(1)  # Make rating predictions in the final layer.
        ])

        # Create the loss function to learn the model with.
        self.__task = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def call(self, data):
        if self.__label_name in data:
            data.pop(self.__label_name)
        embedded = [self.__embeddings[key](data[key]) for key in data.keys()]
        return self.__model(tf.concat(embedded, axis=1))

    def compute_loss(self, data, training=False):
        labels = data.pop(self.__label_name)
        predictions = self(data)
        return self.__task(labels=labels, predictions=predictions)