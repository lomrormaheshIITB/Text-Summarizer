<!-- FILE STRUCTURE -->
1. util.py
2. news.csv
3. train.py
4. README.mb
5. checkpoints
6. inference.py
7. preprocess.py
8. transformer.py
9. extract_data.py
10. summary_tokenizer_pickle
11. document_tokenizer_pickle
12. Report
13. Presentation


<!-- DESCRIPTION -->
1. util.py
	important utility functions

2. news.csv
	dataset

3. train.py
	training model

4. README.md
	README file

5. checkpoints
	directory containing checkpoints

6. inference.py
	test model

7. preprocess.py
	input data preprocessed here

8. transformer.py
	The Model

9. extract_data.py
	extract data from web given url

10. summary_tokenizer_pickle
	stored tokenizer to use in inference

11. document_tokenizer_pickle
	stored tokenizer to use in inference



<!-- ##################################################################### -->
				<!-- *********** TRAINING ********** -->

				remove the checkpoint directory

				command: python3 train.py

				It train the transformer model 


<!-- ##################################################################### -->
				<!-- ************ INFERENCE ********** -->

				command:  python3 inference.py

				1/2 for url/document 

				generate summary for document



<!-- ##################################################################### -->
*** IMPORTANT NOTE ***
1. If you want to use another dataset change the preprocess.py
2. Training takes around 2hour/epoch for me.
3. download dataset, convert to csv format, rename it to news.csv and put in parent directory
4. Before testing train the model once.