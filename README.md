# IDS721-Project5：Predictions for the Operations of Starups
# Introduction
This is the final group project for the course IDS721. We found an interesting feature code competition on Kaggle in June and decided to use it for a systematic exercise. The competition called "Optiver Realized Volatility Prediction" was launched by Optiver, a leading global electronic market maker. This competition lasts for three months and aims to predict short-term volatility for hundreds of stocks across different sectors through the establishment of models. In financial markets, volatility captures the amount of fluctuation in prices. For trading firms like Optiver, accurately predicting volatility is essential for the trading of options, whose price is directly related to the volatility of the underlying product. Optiver’s teams have spent countless hours building sophisticated models that predict volatility and continuously generate fairer options prices for end investors. Optiver wants to take its model to the next level with the help of this competition. 


# Dataset Context
## book_[train/test].parquet 

A parquet file partitioned by stock_id. Provides order book data on the most competitive buy and sell orders entered into the market. The top two levels of the book are shared. The first level of the book will be more competitive in price terms, it will then receive execution priority over the second level.

stock_id - ID code for the stock. Not all stock IDs exist in every time bucket. Parquet coerces this column to the categorical data type when loaded; you may wish to convert it to int8.
time_id - ID code for the time bucket. Time IDs are not necessarily sequential but are consistent across all stocks.
seconds_in_bucket - Number of seconds from the start of the bucket, always starting from 0.
bid_price[1/2] - Normalized prices of the most/second most competitive buy level.
ask_price[1/2] - Normalized prices of the most/second most competitive sell level.
bid_size[1/2] - The number of shares on the most/second most competitive buy level.
ask_size[1/2] - The number of shares on the most/second most competitive sell level.


## trade_[train/test].parquet 

A parquet file partitioned by stock_id. Contains data on trades that actually executed. Usually, in the market, there are more passive buy/sell intention updates (book updates) than actual trades, therefore one may expect this file to be more sparse than the order book.

stock_id - Same as above.
time_id - Same as above.
seconds_in_bucket - Same as above. Note that since trade and book data are taken from the same time window and trade data is more sparse in general, this field is not necessarily starting from 0.
price - The average price of executed transactions happening in one second. Prices have been normalized and the average has been weighted by the number of shares traded in each transaction.
size - The sum number of shares traded.
order_count - The number of unique trade orders taking place.

## train.csv 

The ground truth values for the training set.

stock_id - Same as above, but since this is a csv the column will load as an integer instead of categorical.
time_id - Same as above.
target - The realized volatility computed over the 10 minute window following the feature data under the same stock/time_id. There is no overlap between feature and target data. You can find more info in our tutorial notebook.

## test.csv 

Provides the mapping between the other data files and the submission file. As with other test files, most of the data is only available to your notebook upon submission with just the first few rows available for download.

stock_id - Same as above.
time_id - Same as above.
row_id - Unique identifier for the submission row. There is one row for each existing time ID/stock ID pair. Each time window is not necessarily containing every individual stock.
sample_submission.csv - A sample submission file in the correct format.

row_id - Same as in test.csv.
target - Same definition as in train.csv. The benchmark is using the median target value from train.csv.

# Objective
The objective is to predict whether a startup which is currently operating turns into a success or a failure. The success of a company is defined as the event that gives the company's founders a large sum of money through the process of M&A (Merger and Acquisition) or an IPO (Initial Public Offering). A company would be considered as failed if it had to be shut down.

# About the Data
The data contains industry trends, investment insights and individual company information. There are 48 columns/features. Some of the features are:

agefirstfunding_year – quantitative
agelastfunding_year – quantitative
relationships – quantitative
funding_rounds – quantitative
fundingtotalusd – quantitative
milestones – quantitative
agefirstmilestone_year – quantitative
agelastmilestone_year – quantitative
state – categorical
industry_type – categorical
has_VC – categorical
has_angel – categorical
has_roundA – categorical
has_roundB – categorical
has_roundC – categorical
has_roundD – categorical
avg_participants – quantitative
Is_top500 – categorical
status(acquired/closed) – categorical (the target variable, if a startup is ‘acquired’ by some other organization, means the startup succeed)

# To implement this project

# step1: store data into AWS S3
a. go to AW3 S3 console, click "add bucket"
<img width="1120" alt="6b67a5e08b2cdc5960de721e65aeffc" src="https://user-images.githubusercontent.com/122952572/227016699-04d120c9-862e-4523-9cbe-d3d3c0336a2b.png">
b. upload your file into your bucket
<img width="1120" alt="43acd81ce5570c606712f055059cfa5" src="https://user-images.githubusercontent.com/122952572/227016812-43de35fb-9524-41a0-aa00-7c297941194d.png">

# step2: go to AWS sagemarket and lauch your notebook
a. go to AWS sagemarket console, click "notebook"
b. click "create notebook instance" 
<img width="1075" alt="3d5ca59a60f0f7fa354b150da58434e" src="https://user-images.githubusercontent.com/122952572/227017059-88c77066-1d13-4062-8bcb-37f4ab6be17f.png">
configure your notebook as following
![image](https://user-images.githubusercontent.com/122952572/227017476-4579bc12-2784-4c91-8998-c51755419723.png)

<img width="328" alt="b7af7c68ca98e648da88258a81b8fcd" src="https://user-images.githubusercontent.com/122952572/227017340-2717ed3e-7493-4b71-bfb7-8d30dde58c0f.png">

![image](https://user-images.githubusercontent.com/122952572/227017659-e551c0c9-ffd8-4958-921f-54cbfa79ead9.png)

c. wait a few minutes. In the Notebook instances section, the new SageMaker-Tutorial notebook instance is displayed with a Status of Pending. The notebook is ready when the Status changes to InService.
<img width="1120" alt="78b443a298c7680bb1dacd417a6c0e0" src="https://user-images.githubusercontent.com/122952572/227017854-f9ec3f21-8b6e-4592-80f8-dbebaabc9546.png">


<img width="563" alt="09adc7afdd89deca12a112b1f6d7154" src="https://user-images.githubusercontent.com/122952572/227017905-cf1dabb2-d373-48fa-af5e-0a9a7ed3afbc.png">


d. After your SageMaker-Tutorial notebook instance status changes to InService, choose Open Jupyter.
e. In Jupyter, choose New and then choose conda_python3.

# step3: store data into AWS S3

# step4: load test - locust
We used locust for load teat
a. Install: pip3 install locust
b. Run: locust -f locustfile.py --host=https://7iz1uma2gk.execute-api.us-east-1.amazonaws.com
c. Once you have run it. Go to http://localhost:8089 in your browser and enter Locust's web console like this:





# overall workflow
1. Prepare data (but need to go throght data exploration, data cleaning)
2. Train the ML model
3. Deploy the model
4. Evaluate model performance
5. Clean up (Not terminating your resources will result in charges to your account.)
 
 
 For the full result, please refer to this https://github.com/ht175/IDS721-Project3/blob/main/start_up_prediction.ipynb
