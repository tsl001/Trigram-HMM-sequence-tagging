To run the program first run 

python count_freqs.py gene.train > gene.counts

then run the gene_tagger.py file by running

python gene_tagger.py gene.counts gene.dev > gene_dev.p1.out

then run the eval_gene_tagger.py file to obtain the results

python eval gene tagger.py gene.key gene dev.p1.out