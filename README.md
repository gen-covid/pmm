pmm.py inlcudes the code for selecting the input features, training and testing the model

The code is based on the hypothesis that boolean features as described [1] are available as CSV files in the directory booleans with names:
* data_al1_ultrarare.csv
* data_al2_ultrarare.csv
* data_al1_just_rare.csv
* data_al2_just_rare.csv
* data_al1_just_medium.csv
* data_al2_just_medium.csv
* data_gc_unique_hetero.csv
* data_gc_unique_homo.csv

Phenotype information should also be provided in the form of CSV file with columns age, gender, category; where category defines the clinical category as described in [1]

[1] https://pubmed.ncbi.nlm.nih.gov/34889978/
