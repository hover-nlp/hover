# !/bin/bash
set -eu -o pipefail

mkdir -p data
cd data

wget https://nlp.cs.unc.edu/data/hover/wiki_wo_links.db

mkdir -p hover
cd hover
wget https://hover-nlp.github.io/data/hover/hover_train_release_v1.0.json
wget https://hover-nlp.github.io/data/hover/hover_dev_release_v1.0.json
wget https://hover-nlp.github.io/data/hover/hover_test_release_v1.0.json

mkdir -p doc_retrieval
mkdir -p sent_retrieval
mkdir -p claim_verification

mkdir -p tfidf_retrieved
cd tfidf_retrieved
wget https://nlp.cs.unc.edu/data/hover/train_tfidf_doc_retrieval_results.json
wget https://nlp.cs.unc.edu/data/hover/dev_tfidf_doc_retrieval_results.json
wget https://nlp.cs.unc.edu/data/hover/test_tfidf_doc_retrieval_results.json

cd .. 
cd ..
cd ..