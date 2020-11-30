import argparse
import os
import json
import string
import sqlite3
import collections
import unicodedata
import logging
import nltk


def connect_to_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    return c


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--data_split",
        default=None,
        type=str,
        required=True,
        help="[train | dev | test]",
    )
    parser.add_argument(
        "--sent_retrieve_range",
        default=5,
        type=int,
        help="Top k retrieved documents to be used in sentence selection."
    )
    parser.add_argument(
        "--data_dir",
        default='data',
        type=str,
    )
    parser.add_argument(
        "--dataset_name",
        default='hover',
        type=str
    )
    parser.add_argument(
        "--sent_retrieval_output_dir",
        default="exp1.0",
        type=str,
    )
    parser.add_argument(
        "--sent_retrieval_model_global_step",
        default=1900,
        type=int
    )

    args = parser.parse_args()
    wiki_db = connect_to_db(os.path.join(args.data_dir, 'wiki_wo_links.db'))

    args.data_dir = os.path.join(args.data_dir, args.dataset_name)
    hover_data = json.load(open(os.path.join(args.data_dir, 'hover_'+args.data_split+'_release_v1.1.json')))

    args.sent_retrieval_output_dir = os.path.join('out', args.dataset_name, args.sent_retrieval_output_dir, 'sent_retrieval', \
        'checkpoint-'+str(args.sent_retrieval_model_global_step))
    sent_retrieval_predictions = json.load(open(os.path.join(args.sent_retrieval_output_dir, args.data_split+'_predictions_.json')))
    sent_retrieval_data = json.load(open(os.path.join(args.data_dir, 'sent_retrieval', 'hover_'+args.data_split+'_sent_retrieval.json')))

    uid_to_wikidocuments = {}
    for e in sent_retrieval_data:
        uid, claim, context = e['id'], e['claim'], e['context']
        assert uid not in uid_to_wikidocuments
        uid_to_wikidocuments[uid] = [claim, context]

    uid_to_label = {}
    for e in hover_data:
        uid, label = e['uid'], e['label']
        assert uid not in uid_to_label
        uid_to_label[uid] = label

    data_for_claim_verif = []
    for uid in sent_retrieval_predictions.keys():
        pred_obj = sent_retrieval_predictions[uid]
        predicted_sp = pred_obj["predicted_sp"]

        claim, wikidocuments = uid_to_wikidocuments[uid]
        wiki_titles_to_documents = {}
        for _title, _doc in wikidocuments:
            wiki_titles_to_documents[_title] = _doc

        context = []
        for _doc_title, sent_id in predicted_sp:
            sent = wiki_titles_to_documents[_doc_title][sent_id]
            context.append(sent)

        # print(predicted_sp)
        # print(wiki_titles_to_documents)
        # print(context)
        # exit()
        context = ' '.join(context)
        label = uid_to_label[uid]
        dp = {'id': uid, 'claim': claim, 'context': context, 'label': label}
        data_for_claim_verif.append(dp)
    

    logging.info("Saving prepared data ...")
    with open(os.path.join(args.data_dir, 'claim_verification', 'hover_'+args.data_split+'_claim_verification.json'), 'w', encoding="utf-8") as f:
        json.dump(data_for_claim_verif, f)


if __name__ == "__main__":
    main()