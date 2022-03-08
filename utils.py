import re
import math
import pandas as pd
import numpy as np


# Replace NaN values with a default value appropriate for the data type
def fill_nan(data):
    for key, dtype in zip(data.keys(), data.dtypes):
        default = 'missing' if dtype == 'object' else 0
        data[key] = data[key].fillna(default)

        if dtype == 'object':
            data[key] = data[key].replace({'': default})
    return data


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


def calc_euclid_dist(x, y):
    return math.sqrt(sum([(x_val - y_val) ** 2 for x_val, y_val in zip(x, y)]))


def map_ranking(ranking):
    return [{
        'title': r[0],
        'gender': r[1],
        'rank': i + 1,
        'score': r[2],
    } for i, r in enumerate(ranking)]


def calc_metrics_and_zscore(mapped_ranking, original_data, sample=False):
    metrics = []
    r2_start = 1
    for r in mapped_ranking:
        for i in range(r2_start, len(mapped_ranking)):
            r2 = mapped_ranking[i]
            r_full = original_data[original_data['title'] == r['title']].copy()
            r2_full = original_data[original_data['title'] == r2['title']].copy()

            r_title = r_full.pop('title').values[0]
            r2_title = r2_full.pop('title').values[0]

            r_gender = r_full.pop('author_gender').values[0]
            r2_gender = r2_full.pop('author_gender').values[0]

            r_birthplace = r_full.pop('birthplace').values[0]
            r2_birthplace = r2_full.pop('birthplace').values[0]

            # If genders or birthplace are different, then assign one individual a value so the
            # difference can be counted. What is a good value to encode the difference in gender and birthplace?
            if r_gender != r2_gender:
                r_full['author_gender'] = 1
                r2_full['author_gender'] = 0
            else:
                r_full['author_gender'] = 0
                r2_full['author_gender'] = 0
            if r_birthplace != r2_birthplace:
                r_full['birthplace'] = 1
                r2_full['birthplace'] = 0
            else:
                r_full['birthplace'] = 0
                r2_full['birthplace'] = 0

            individual_dist = calc_euclid_dist(
                r_full.values.flatten().tolist(),
                r2_full.values.flatten().tolist()
            )

            # eval_dist = abs(r['rank'] - r2['rank'])
            eval_dist = calc_euclid_dist([r['rank']], [r2['rank']])

            metrics.append({
                'book1': {
                    'title': r_title,
                    'gender': r_gender,
                    'birthplace': r_birthplace
                },
                'book2': {
                    'title': r2_title,
                    'gender': r2_gender,
                    'birthplace': r2_birthplace
                },
                'individual_dist': individual_dist,
                'eval_dist': eval_dist
            })
        r2_start += 1

    # Calc mean and standard deviation for individual and eval metrics for normalization
    metrics_len = len(metrics)
    all_indiv_metrics = [m['individual_dist'] for m in metrics]
    all_eval_metrics = [m['eval_dist'] for m in metrics]

    indiv_mean = sum(all_indiv_metrics) / metrics_len
    eval_mean = sum(all_eval_metrics) / metrics_len

    n = metrics_len - 1 if sample else metrics_len
    indiv_std_dev = math.sqrt(sum([(v - indiv_mean) ** 2 for v in all_indiv_metrics]) / n)
    indiv_std_dev = indiv_std_dev if indiv_std_dev != 0 else 1
    eval_std_dev = math.sqrt(sum([(v - eval_mean) ** 2 for v in all_eval_metrics]) / n)
    eval_std_dev = eval_std_dev if eval_std_dev != 0 else 1

    # z score all and take absolute value since for individual fairness we're really supposed to be comparing distance
    # Individual z score represents how "different" 2 individuals are compared to all other pairs of individuals
    # Eval z score represents how "differently" the individual was ranked compared to other individuals
    satisfy_indiv_fairness = 0
    for m in metrics:
        m['z_indiv_dist'] = abs((m['individual_dist'] - indiv_mean) / indiv_std_dev)
        m['z_eval_dist'] = abs((m['eval_dist'] - eval_mean) / eval_std_dev)

        satisfy_indiv_fairness += 1 if m['z_eval_dist'] <= m['z_indiv_dist'] else 0

    return metrics, satisfy_indiv_fairness / metrics_len


def calc_sensitive_attr_ratio(data, original_data):
    genders_in_data = ['male', 'female']
    data_len = len(data)
    merged = pd.merge(pd.DataFrame(data), original_data, on='title')

    gender_num = merged['author_gender'].value_counts()
    birthplace_num = merged['birthplace'].value_counts()

    gender_dict = {g: gender_num[g] / data_len if g in gender_num else 0 for g in genders_in_data}
    birthplace_dict = {b: birthplace_num[b] / data_len for b in birthplace_num.keys()}
    return {
        **gender_dict,
        'birthplace': {
            **birthplace_dict
        }
    }


def rerank(top_dict, original_data):
    merged = pd.merge(pd.DataFrame(top_dict['ranking']), original_data, on='title')
    genders_in_data = ['male', 'female']

    #         Male  Female
    a_upper = [0.9, 0.9]  # Upper bound for range in which fairness should be
    b_lower = [0.3, 0.65]  # Lower bound for range in which fairness should be
    l_groups = 2
    a_min = max(a_upper)
    a_max = min(a_upper)
    l_star = b_lower.index(min(b_lower))
    N = len(top_dict['ranking'])
    k = N

    e = (2 / k) * max([
        1 + (l_groups / sum([a_upper[L] - b_lower[L] for L in range(l_groups)])),
        1 + (l_groups / (1 - sum(b_lower))),
        max([1 + (2 / (a_upper[L] - b_lower[L])) for L in range(l_groups)])
    ])
    B = math.floor(e * k / 2)
    b = min([
        math.floor(a_min * B),
        B - sum([math.ceil(b_lower[L] * B) if L != l_star else 0 for L in range(l_groups)])
    ])
    M = math.ceil(N / b) * B

    def recurse_edit(data, new_index, old_index):
        if data[new_index] is not None:
            recurse_edit(data, new_index + 1, new_index)
        data[new_index] = data[old_index]
        data[old_index] = None

    new_rank = [top_dict['ranking'][i] if i < 30 else None for i in range(1000)]

    for i in range(math.ceil(N / b), -1, -1):
        for j in range(1, min(b, N - (i - 1))):
            old = b * (i - 1) + j
            new = (i - 1) * B + j
            # print(f'Move item at {old} to {new}')
            recurse_edit(new_rank, new, old)

    for j in range(M):
        if new_rank[j] is None:
            i = math.ceil(j / B)
            for j2 in range(j + 1, M):
                if new_rank[j2] is not None:
                    group = genders_in_data.index(
                        merged[merged['title'] == new_rank[j2]['title']]['author_gender'].values[0]
                    )

                    # For group, check if the lower bound is satisfied by the fairness result for the ith block
                    #  a block contains the items at the ranks [(i - 1) * k + 1, (i - 1) * k + 2, ..., (i - 1) * k + k]
                    # OR
                    # Check if lower bound for all groups satisfied (for all groups, fairness is above lower bound)
                    #  AND
                    # Check if upper bound for group wouldn't be violated ()
                    # Then move item at rank j2 to rank j, then break
                    satis = top_dict['satisfaction_rate']

                    if satis < b_lower[group] or (b_lower[0] < satis < a_upper[group] and b_lower[1] < satis):
                        recurse_edit(new_rank, j, j2)
                        break

    for j in range(N):
        if new_rank[j] is None:
            for i in range(j - 1, -1, -1):
                if new_rank[i] is not None:
                    new_rank[j] = new_rank[i]
                    new_rank[i] = None
                    break

    new_rank = list(filter(lambda x: x is not None, new_rank))
    out_of_order_ranks = [r['rank'] for r in new_rank]
    out_of_order_scores = [r['score'] for r in new_rank]
    out_of_order_ranks.sort()
    out_of_order_scores.sort()
    ordered_ranks = out_of_order_ranks
    ordered_scores = list(reversed(out_of_order_scores))
    return [{
        'title': out_of_order['title'],
        'gender': out_of_order['gender'],
        'rank': ordered_r,
        'score': ordered_s
    } for out_of_order, ordered_r, ordered_s in zip(new_rank, ordered_ranks, ordered_scores)]


def calc_individual_fairness_metrics_and_accuracy(prediction, original_data, top_only=False, top=10):
    top_ranked = map_ranking(prediction[:top])

    top_metrics, top_satisfy_rate = calc_metrics_and_zscore(top_ranked, original_data, sample=True)
    top_dict = {
        'ranking': top_ranked,
        'metrics': top_metrics,
        'satisfaction_rate': top_satisfy_rate,
        **calc_sensitive_attr_ratio(top_ranked, original_data)
    }

    new_rank = rerank(top_dict, original_data)
    new_top_metrics, new_top_satisfy_rate = calc_metrics_and_zscore(new_rank, original_data, sample=True)
    new_top_dict = {
        **{f'old_{k}': top_dict[k] for k in top_dict.keys()},
        **calc_sensitive_attr_ratio(top_ranked, original_data),
        'new_ranking': new_rank,
        'new_metrics': new_top_metrics,
        'new_satisfaction_rate': new_top_satisfy_rate,
        'improved': new_top_satisfy_rate > top_satisfy_rate
    }

    if top_only:
        return new_top_dict

    # all_metrics, all_satisfy_rate = calc_metrics_and_zscore(mapped_ranking, original_data)
    #
    # return {
    #     'ranking': mapped_ranking,
    #     'metrics': all_metrics,
    #     'satisfaction_rate': all_satisfy_rate,
    #     **calc_sensitive_attr_ratio(mapped_ranking, original_data)
    # }, top_dict
