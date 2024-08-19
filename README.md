### Experiment 
+ Measure how often the Conversationl AI correctly avoids answering the irrelevant queries

### Methodology: Training a machine learning model that:-
+ Label "AI FAIL" : AI responds to an irrelevant query
+ Label "AI PASS" : AI appropriately avoids an irrelevant query

### Dataset 
+ Provide 8 categories of irrelevant responses, each containing 5 examples, totaling 40 inputs. Descriptive answers in this context will be classified as "AI FAIL" 
+ 40 inputs of appropriate AI responses to irrelevant questions. Responses like these will be classified as "AI PASS"

### Test Conditions  
+ Logistic Regression is used as the classification algorithm.
+ It contains two columns: 'answer' (the text data) and 'MU_score' (the label).
+ TfidfVectorizer (Term Frequency-Inverse Document Frequency) is used for feature extraction from the text data. 
+ The test size is set to 0.2 (20% of the data), with a random state of 3 for reproducibility.
Accuracy score is used to evaluate the model's performance on both training and test data.

### Next Steps 
+ Increase the size of training and testing dataset
+ ML algorithm is prone to errors itself
+ Unaware of the boundary conditions
+ Training the model on irrelevant responses feels pointless, as the model only learns that responses in this category are labeled "AI FAIL" but doesn't grasp the meaning of "AI FAIL".
A potential solution could be to classify responses as "AI PASS" if they fit a certain criterion; otherwise, label as "AI FAIL".
+ Investigate how guardrails function, as they assess whether queries are relevant or not. While we focuses on the answer side, guardrails address the query side. Both aim to determine the relevance of something.
    
