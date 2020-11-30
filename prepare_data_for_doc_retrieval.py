import argparse
import os
import json
import string
import sqlite3
import collections
import unicodedata
import logging


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
        "--doc_retrieve_range",
        default=20,
        type=int,
        help="Top k tfidf-retrieved documents to be used in neural document retrieval."
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
        "--oracle",
        action="store_true"
    )

    args = parser.parse_args()
    wiki_db = connect_to_db(os.path.join(args.data_dir, 'wiki_wo_links.db'))

    args.data_dir = os.path.join(args.data_dir, args.dataset_name)

    hover_data = json.load(open(os.path.join(args.data_dir, 'hover_'+args.data_split+'_release_v1.1.json')))
    tfidf_retrieved_doc = json.load(open(os.path.join(args.data_dir, 'tfidf_retrieved', args.data_split+'_tfidf_doc_retrieval_results.json')))

    uid_to_tfidf_retrieved_doc = {}
    for e in tfidf_retrieved_doc:
        uid = e['id']
        assert uid not in uid_to_tfidf_retrieved_doc
        uid_to_tfidf_retrieved_doc[uid] = e['doc_retrieval_results'][0][0]

    hover_data_w_tfidf_docs = []

    for e in hover_data:
        uid, num_hops, supporting_facts = e['uid'], e['num_hops'], e['supporting_facts']
        retrieved_docs = uid_to_tfidf_retrieved_doc[uid]
        golden_docs = []
        for sp in supporting_facts:
            if sp[0] not in golden_docs:
                golden_docs.append(sp[0])

        context, labels = [], []

        if args.oracle:
            for doc_title in golden_docs:
                para = wiki_db.execute("SELECT * FROM documents WHERE id=(?)", \
                                            (unicodedata.normalize('NFD', doc_title),)).fetchall()[0]
                context.append(list(para))
                labels.append(1)

        for doc_title in retrieved_docs[:20]:
            if args.oracle and doc_title in golden_docs:
                continue
            para = wiki_db.execute("SELECT * FROM documents WHERE id=(?)", \
                                            (unicodedata.normalize('NFD', doc_title),)).fetchall()[0]

            context.append(list(para))
        
            if doc_title in golden_docs:
                labels.append(1)
            else:
                labels.append(0)
        e['context'] = context[:20]
        e['labels'] = labels[:20]
        hover_data_w_tfidf_docs.append(e)

    logging.info("Saving prepared data ...")
    if args.oracle:
        with open(os.path.join(args.data_dir, 'doc_retrieval', 'hover_'+args.data_split+'_doc_retrieval_oracle.json'), 'w', encoding="utf-8") as f:
            json.dump(hover_data_w_tfidf_docs, f)
    else:
        with open(os.path.join(args.data_dir, 'doc_retrieval', 'hover_'+args.data_split+'_doc_retrieval.json'), 'w', encoding="utf-8") as f:
            json.dump(hover_data_w_tfidf_docs, f)


if __name__ == "__main__":
    main()