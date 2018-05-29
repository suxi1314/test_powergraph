#! /bin/sh
cd /graphlab/release/toolkits/topic_modeling
./cgs_lda --corpus ./daily_kos/tokens --dictionary ./daily_kos/dictionary.txt --word_dir word_counts --doc_dir doc_counts --burnin=60 
