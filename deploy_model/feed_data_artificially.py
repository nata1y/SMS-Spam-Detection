import copy

import pandas as pd
from joblib import load

from deploy_model.proccess_stats import compare_nlp_models, compare_loss_dist
from train_model.text_preprocessing import prepare, _text_process, _extract_message_len

from deploy_model.serve_model import load_best_clf, classifier_name, stats as general_stats


def main():
    stats = general_stats
    data = pd.read_csv(
        'dataset/SMSSpamCollection',
        sep='\t',
        names=['label', 'message']
    )
    dt = 'train_data'
    model = load_best_clf()
    losses = []
    for idx, row in data.iterrows():
        print(idx)
        processed_sms = prepare(row['message'])
        prediction = model.predict(processed_sms)[0]

        stats = stats.append({
            "result": prediction,
            "prob_spam": model.predict_proba(processed_sms)[0],
            "classifier": classifier_name,
            "sms": row['message']
        }, ignore_index=True)

        losses.append(0.0 if prediction == row['label'] else 1.0)

        stats.to_csv('output/stats/stats_from_wild.csv', index=False)

        if (stats.shape[0]) % 1000 == 0:
            compare_nlp_models(stats["sms"].tolist()[-1000:], dt)
            # compare_loss_dist(losses, dt)
            losses = []


if __name__ == '__main__':
    main()
