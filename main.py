import math

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from utils import preprocess_all_data, calc_individual_fairness_metrics_and_accuracy


if __name__ == '__main__':
    # preprocess_all_data()
    data = pd.read_csv('data/preprocessed/merged.csv').sample(frac=1)
    book_gendered = pd.read_csv('data/preprocessed/books_sensitive.csv').drop_duplicates(subset=['title'])

    # Set up data in sensitive book data
    recommend_author = book_gendered.pop('author')

    # Used later to compute distance metrics between recommendations
    copied_gender_data = book_gendered.copy()

    num_ratings = []
    user_ids = []

    recommend_birthplace = book_gendered.pop('birthplace')
    recommend_gender = book_gendered.pop('author_gender')
    recommend_title = book_gendered.pop('title')
    gender_and_title = pd.concat([recommend_title, recommend_gender], axis=1)

    old_individual_fairness_satisfaction_rates = []
    new_individual_fairness_satisfaction_rates = []
    replaced_worst_individual_fairness_satisfaction_rates = []

    fairness_improved = []
    fairness_improved_by = []

    male_appearance_rates = []
    female_appearance_rates = []
    birthplace_appearance_rates = {}
    avg_birthplace_appearance_rates = {}

    old_male_scores = []
    old_male_ranks = []
    old_female_scores = []
    old_female_ranks = []

    new_male_scores = []
    new_male_ranks = []
    new_female_scores = []
    new_female_ranks = []

    replaced_male_scores = []
    replaced_male_ranks = []
    replaced_female_scores = []
    replaced_female_ranks = []

    # TODO final run use 50
    users_with_x_or_more_ratings = [
        user for user in data['user_id'].unique()
        if len(data[data['user_id'] == user]) >= 50
    ]
    print(f'Learning recommendation ranker for {len(users_with_x_or_more_ratings)} users...')

    for user in users_with_x_or_more_ratings:
        user_data = data[data['user_id'] == user].copy()
        user_data.drop(['user_id'], axis=1, inplace=True)

        num_ratings.append(len(user_data))
        user_ids.append(user)

        # Just using user_data as the training, no splitting for validation since there are so few ratings per person.
        # Just going to validate on the sensitive data.
        train_y = user_data.pop('rating')
        train_authors = user_data.pop('author')
        train_titles = user_data.pop('title')
        data_feat_num = user_data.shape[1]

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(data_feat_num, activation='relu', input_shape=(data_feat_num,)),
            tf.keras.layers.Dense(math.floor(data_feat_num / 4)),
            tf.keras.layers.Dense(math.floor(data_feat_num / 16), activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor='loss', patience=600, min_delta=0.0001, restore_best_weights=True)
        ]
        model.compile(optimizer=tf.optimizers.Adam(learning_rate=0.001),
                      loss=tf.keras.losses.MeanAbsoluteError())

        # TODO final run use epoch=8000
        train_hist = model.fit(user_data, train_y, callbacks=callbacks, epochs=8000, verbose=0)

        # Getting a prediction using gendered data
        test_pred = [v[0] for v in model.predict(book_gendered)]

        # ranking_test = [(t, p, norm) for t, p, norm in zip(recommend_title, test_pred, z_scored_rank)]
        ranking_test = [(t, g, p) for t, g, p in zip(recommend_title, recommend_gender, test_pred)]
        ranking_test.sort(key=lambda x: x[2])
        ranking_test = list(reversed(ranking_test))

        # all_ranked, top = calc_individual_fairness_metrics_and_accuracy(ranking_test, copied_gender_data)
        # print(all_ranked['satisfaction_rate'])

        top_num = 30
        top = calc_individual_fairness_metrics_and_accuracy(ranking_test, copied_gender_data,
                                                            top_only=True, top=top_num)

        for r in top['old_ranking']:
            if r['gender'] == 'male':
                old_male_ranks.append(r['rank'])
                old_male_scores.append(r['score'])

                if not top["improved"]:
                    replaced_male_ranks.append(r['rank'])
                    replaced_male_scores.append(r['score'])
            if r['gender'] == 'female':
                old_female_ranks.append(r['rank'])
                old_female_scores.append(r['score'])
                if not top["improved"]:
                    replaced_female_ranks.append(r['rank'])
                    replaced_female_scores.append(r['score'])

        for r in top['new_ranking']:
            if r['gender'] == 'male':
                new_male_ranks.append(r['rank'])
                new_male_scores.append(r['score'])
                if top["improved"]:
                    replaced_male_ranks.append(r['rank'])
                    replaced_male_scores.append(r['score'])
            if r['gender'] == 'female':
                new_female_ranks.append(r['rank'])
                new_female_scores.append(r['score'])
                if top["improved"]:
                    replaced_female_ranks.append(r['rank'])
                    replaced_female_scores.append(r['score'])

        print(top['old_satisfaction_rate'])
        print(top['new_satisfaction_rate'])
        print(f'Improved? {top["improved"]}')
        print(f'Printing top 10 of {top_num} new ranked:')
        print(top['new_ranking'][:10])
        print()

        old_individual_fairness_satisfaction_rates.append(top['old_satisfaction_rate'])
        new_individual_fairness_satisfaction_rates.append(top['new_satisfaction_rate'])
        replaced_worst_individual_fairness_satisfaction_rates.append(
            top['new_satisfaction_rate'] if top["improved"] else top['old_satisfaction_rate']
        )
        fairness_improved.append(top['improved'])
        fairness_improved_by.append(top['new_satisfaction_rate'] - top['old_satisfaction_rate']
                                    if top['improved'] else 0)

        male_appearance_rates.append(top['male'])
        female_appearance_rates.append(top['female'])

        for b in top['birthplace'].keys():
            if b not in birthplace_appearance_rates:
                birthplace_appearance_rates[b] = []
            birthplace_appearance_rates[b].append(top['birthplace'][b])

        # # Get predicted ranking for training data. Not really used in project but interesting to see.
        # train_pred = model.predict(user_data)
        # ranking_train = [(t, y, p[0]) for t, y, p in zip(train_titles, train_y, train_pred)]
        # ranking_train.sort(key=lambda x: x[2])
        # ranking_train = list(reversed(ranking_train))
        # print('\nPrinting ranking of training data:')
        # print(ranking_train)
        # print()

    num_results = len(old_individual_fairness_satisfaction_rates)
    old_avg_individual_fairness_satisfaction_rate = np.average(old_individual_fairness_satisfaction_rates)
    new_avg_individual_fairness_satisfaction_rate = np.average(new_individual_fairness_satisfaction_rates)
    replaced_worst_avg_individual_fairness_satisfaction_rate = np.average(
        replaced_worst_individual_fairness_satisfaction_rates
    )
    avg_female_appearance_rate = np.average(female_appearance_rates)
    avg_male_appearance_rate = np.average(male_appearance_rates)
    fair_filtered = list(filter(lambda x: x != 0, fairness_improved_by))
    avg_improved_by = np.average(fair_filtered)

    avg_old_male_scores = np.average(old_male_scores)
    avg_old_male_ranks = np.average(old_male_ranks)
    avg_old_female_scores = np.average(old_female_scores)
    avg_old_female_ranks = np.average(old_female_ranks)

    avg_new_male_scores = np.average(new_male_scores)
    avg_new_male_ranks = np.average(new_male_ranks)
    avg_new_female_scores = np.average(new_female_scores)
    avg_new_female_ranks = np.average(new_female_ranks)

    avg_replaced_male_scores = np.average(replaced_male_scores)
    avg_replaced_male_ranks = np.average(replaced_male_ranks)
    avg_replaced_female_scores = np.average(replaced_female_scores)
    avg_replaced_female_ranks = np.average(replaced_female_ranks)

    did_improve_num = fairness_improved.count(True)
    fairness_improved_rate = did_improve_num / len(fairness_improved)
    test_fair_improve_rate = np.average(did_improve_num)

    for b in birthplace_appearance_rates.keys():
        if b not in avg_birthplace_appearance_rates:
            avg_birthplace_appearance_rates[b] = []
        avg_birthplace_appearance_rates[b] = [np.average(birthplace_appearance_rates[b])if i == 0 else 0
                                              for i in range(len(old_individual_fairness_satisfaction_rates))]

    pd.DataFrame({
        'user_id': user_ids,
        'number_of_ratings': num_ratings,
        'old_indiv_satisfaction_rate': old_individual_fairness_satisfaction_rates,
        'old_avg_indv_satisfaction_rate': [old_avg_individual_fairness_satisfaction_rate
                                           if i == 0 else 0
                                           for i in range(num_results)],
        'new_indiv_satisfaction_rate': new_individual_fairness_satisfaction_rates,
        'new_avg_indv_satisfaction_rate': [new_avg_individual_fairness_satisfaction_rate
                                           if i == 0 else 0
                                           for i in range(num_results)],
        'replaced_worst_avg_individual_fairness_satisfaction_rate': [
            replaced_worst_avg_individual_fairness_satisfaction_rate
            if i == 0 else 0
            for i in range(num_results)
        ],
        'did_fairness_improve': fairness_improved,
        'fairness_improved_rate': [fairness_improved_rate if i == 0 else 0 for i in range(num_results)],
        'improved_by': fairness_improved_by,
        'avg_improved_by': [avg_improved_by if i == 0 else 0 for i in range(num_results)],
        'male_appearance_rates': male_appearance_rates,
        'avg_male_appearance_rate': [avg_male_appearance_rate if i == 0 else 0 for i in range(num_results)],
        'female_appearance_rates': female_appearance_rates,
        'avg_female_appearance_rates': [avg_female_appearance_rate if i == 0 else 0 for i in range(num_results)],
        'avg_old_male_scores': [avg_old_male_scores if i == 0 else 0 for i in range(num_results)],
        'avg_old_male_ranks': [avg_old_male_ranks if i == 0 else 0 for i in range(num_results)],
        'avg_old_female_scores': [avg_old_female_scores if i == 0 else 0 for i in range(num_results)],
        'avg_old_female_ranks': [avg_old_female_ranks if i == 0 else 0 for i in range(num_results)],
        'avg_new_male_scores': [avg_new_male_scores if i == 0 else 0 for i in range(num_results)],
        'avg_new_male_ranks': [avg_new_male_ranks if i == 0 else 0 for i in range(num_results)],
        'avg_new_female_scores': [avg_new_female_scores if i == 0 else 0 for i in range(num_results)],
        'avg_new_female_ranks': [avg_new_female_ranks if i == 0 else 0 for i in range(num_results)],
        'avg_replaced_male_scores': [avg_replaced_male_scores if i == 0 else 0 for i in range(num_results)],
        'avg_replaced_male_ranks': [avg_replaced_male_ranks if i == 0 else 0 for i in range(num_results)],
        'avg_replaced_female_scores': [avg_replaced_female_scores if i == 0 else 0 for i in range(num_results)],
        'avg_replaced_female_ranks': [avg_replaced_female_ranks if i == 0 else 0 for i in range(num_results)],
        **avg_birthplace_appearance_rates
    }).to_csv('data/fairness_out.csv', index=False)

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

