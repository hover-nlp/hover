# !/bin/bash
set -eu -o pipefail

cd data

wget https://nlp.cs.unc.edu/data/hover/wiki_wo_links.db

cd hover

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