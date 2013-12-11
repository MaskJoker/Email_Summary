Email_Summary
=============

Email Summary NLP project @ NYU

#TODO:
	(1)feature extraction
		f_centroid_similarity
		f_local_centroid_similarity
		
#Note
	(1)If you have changed the database table structure ,you have to run load_bc3_corpus() in main() at main.py
	(2)If you don't need speech act tag for now,just comment load_generated_speech_act_tag() in main() at main.py. load_generated_speech_act_tag() usually takes up to 3 minutes to finish
		

#Functionality
	(1)Import BC3 corpus xml files into sqlite database
	(2)Feature extraction
		2.1)Generate speech act tag(Ddata,Deliver,Meet,Request,Commit) for each email sentences using speech_act.jar(https://github.com/KevinZhaoNY/SpeechActTagger)
		2.2)Sentiment Analysis (give a score between -3 and 3 to each sentence)
		2.3)Sentence Length
	(3)TODO:Export sentence feature matrix to supervised machine learning algorithm

#How to use
main.py will be the main entry of our program where I put three function calls 


    (1)load_bc3_corpus(): 
    	a)read /bc3/corpus.xml
    	b)insert the data into sqlite database
    	
    (2)load_generated_speech_act_tag(): 
    	a)call /libs/speech_act.jar to generate speech act tags
    	b)update sa_tag column in the "sentence" table

    (3)feature_extraction():
    	a)Read data from database
    	b)TODO: do feature extraction
    	c)insert the data into "feature" table,which would be the output to model learning module
    	
NOTE:
	Once the database file "bc3.db" has been created,you don't have to run load_bc3_corpus() and load_generated_speech_act_tag() again,you could simply comment them.
	After you change the database table structure,you should run these two functions again to make the changes into the database. 

#Program Structure
	/bc3/annotation.xml 
	--- email summary for bc3 corpus
	/bc3/corpus.xml 
	--- bc3 email corpus
	/bc3/bc3/db 
	--- bc3 email corpus database file (generated automatically after running main.py)
	
	/docs 
	-- documents and design graphs
	
	/libs/speech_act.jar 
	--- java program which generates speech act tag ,here is the link https://github.com/KevinZhaoNY/SpeechActTagger
	
	/src/main.py  
	-- this is the main entry of the program
	/src/db_tester.py 
	-- shows you all the content in the sqlite database
	/src/sentiment/analysis.py 
	-- sentiment analysis module
	/src/sentiment/AFINN-111.txt 
	-- AFINN: A new word list for sentiment analysis on Twitter
	/src/sentiment/topia 
	--- Content Term Extraction using POS Tagging

#Implementation Note
	(1)Subject similarity @author:Kevin
		a)input: subject_string,sentence_string
		b)output: from 0 to 1 in float
		c)Perform bigram comparison between two strings and return a percentage match in decimal form
			similarity = [2*(matches)]/[len(subject_string)+(sentence_string)]
	(2)Speech act tag @author:Kevin
		
	(3)TFIDF @author:Luming
	(4)......
#DataBase Design

	* thread 
	primary key: id
	-----------------
	|id    |Subject |
	-----------------
	|int   |char    |
	---------------- -  
	
	
	* email
	primary key: id +thread_id
	----------------------------------------------------------------
	|id    |thread_id |subject |from_who |to_whom |cc  |num_replies|
	----------------------------------------------------------------
	|int   |int       |char    |char     |char    |char|int		   |
	----------------------------------------------------------------
	
	* sentence 
	primary key: id+email_id+thread_id
	---------------------------------------------------------------------------------
	|id    |email_id |thread_id|text |length |similarity |extracted|sa_tag|sentiment|
	---------------------------------------------------------------------------------
	|int   |int      |int      |char |int    |float      |boolean  |char  |int      |
	---------------------------------------------------------------------------------
	
	
	* summary  (not implemented yet)
	--------------
	|id    |text |
	--------------
	|int   |char |
	-------------- 
	
	* feature  (not implemented yet) @Luming Nie: you should put your feature value into this table
	primary key: sentence_id+email_id+thread_id
	---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	|sentence_id|email_id|thread_id    |extracted |f_length |f_sentiment |f_thread_line_number |f_relative_thread_line_num|f_centroid_similarity|f_local_centroid_similarity|f_tfidf_sum|f_tfidf_avg|f_email_number|f_relative_email_number|f_subject_similarity|f_reply_number|f_recipients_number|f_sa_tag|
	---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	|int   	    |int     |int      	   |boolean   |int      |int         |int                  |float         	      |int         	    |int         		|int        |int        |int           |int                    |float         	    |int           |int                |int     |    
	---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
	
