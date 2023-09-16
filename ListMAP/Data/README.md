# Dataset 
we use the following six benchmark datasets: MQ2007 and MQ2008 of the Letor 4.0 benchmark (Qin
& Liu, 2013), Set 1 and Set 2 of the Yahoo! learning to rank challenge data set (Chapelle & Chang, 2011), denoted as Yahoo!Set1
and Yahoo!Set2 in our reported results, and Microsoft 30k and Microsoft 10K datasets (Qin & Liu, 2013) denoted as MSLR30K
and MSLR10K, respectively. Table 2 shows the number of queries, documents, and features for each data set. The documents in
MQ2007 and MQ2008 are retrieved from 25 million pages in the Gov2 web page collection (Qin, Liu, Xu et al., 2010) for queries in
the Million Query track of TREC 2007 and TREC 2008, respectively. Documents are labeled with relevance judgment ranging from
0 (not related) to 2 (highly related). Yahoo! data sets consists of top documents retrieved for randomly sampled queries from the
query logs of the Yahoo! search engine. Documents are labeled with relevance judgment ranging from 0 (not related) to 5 (highly
related). MSLR30K is created from a retired labeling set of the Bing search engine, and contains 125 documents for each query by
average. MSLR10K is created by a random sub-sampling from MSLR30K with 10,000 queries. Similar to Yahoo!Sets, the relevance
judgments in MSLR datasets range from 0 (irrelevant) to 5 (perfectly relevant). We used 50 percent of the training data for setting
the 𝛼 and 𝛽 parameters of the prior distribution models and use the rest for the training ListMAP learning to rank models.
MSLR datasets and MQ2007, and MQ2008 comes with five folds, where each fold has a test, train and a validation set. We used
the train set of each fold for training the models, and report the average results across test sets of all folds. Yahoo!Sets comes with
one set of test, train, and validation data and the reported results are those obtained from running model on the test set.


