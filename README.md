# IDS721-Project5：Optiver Realized Volatility Prediction
# Introduction
This is the final group project for the course IDS721. We found an interesting feature code competition on Kaggle in June and decided to use it for a systematic exercise. The competition called "Optiver Realized Volatility Prediction" was launched by Optiver, a leading global electronic market maker. This competition lasts for three months and aims to predict short-term volatility for hundreds of stocks across different sectors through the establishment of models. In financial markets, volatility captures the amount of fluctuation in prices. For trading firms like Optiver, accurately predicting volatility is essential for the trading of options, whose price is directly related to the volatility of the underlying product. Optiver’s teams have spent countless hours building sophisticated models that predict volatility and continuously generate fairer options prices for end investors. Optiver wants to take its model to the next level with the help of this competition. 



# Objective
The project aims to create models that can accurately predict short-term volatility for multiple stocks across different sectors. In financial markets, volatility measures the degree of fluctuation in prices, and predicting it is crucial for trading firms that deal with options. Options prices are closely linked to the volatility of the underlying asset, and accurate volatility predictions are necessary to generate fair options prices for end investors.


To develop effective models, it is essential to consider various factors that can influence stock prices, such as market trends, economic indicators, company-specific news, and trading volumes. You will also need to choose appropriate statistical methods, such as time series analysis, stochastic modeling, and machine learning algorithms, to analyze historical data and forecast future volatility. Furthermore, your models must generate fair and precise options prices that reflect the expected volatility of the underlying stocks. This means that you will need to continuously monitor and update your models based on new data and market conditions.

# Dataset Context
### book_[train/test].parquet 

A parquet file partitioned by stock_id. Provides order book data on the most competitive buy and sell orders entered into the market. The top two levels of the book are shared. The first level of the book will be more competitive in price terms, it will then receive execution priority over the second level.

stock_id - ID code for the stock. Not all stock IDs exist in every time bucket. Parquet coerces this column to the categorical data type when loaded; you may wish to convert it to int8.
time_id - ID code for the time bucket. Time IDs are not necessarily sequential but are consistent across all stocks.
seconds_in_bucket - Number of seconds from the start of the bucket, always starting from 0.
bid_price[1/2] - Normalized prices of the most/second most competitive buy level.
ask_price[1/2] - Normalized prices of the most/second most competitive sell level.
bid_size[1/2] - The number of shares on the most/second most competitive buy level.
ask_size[1/2] - The number of shares on the most/second most competitive sell level.


### trade_[train/test].parquet 

A parquet file partitioned by stock_id. Contains data on trades that actually executed. Usually, in the market, there are more passive buy/sell intention updates (book updates) than actual trades, therefore one may expect this file to be more sparse than the order book.

stock_id - Same as above.
time_id - Same as above.
seconds_in_bucket - Same as above. Note that since trade and book data are taken from the same time window and trade data is more sparse in general, this field is not necessarily starting from 0.
price - The average price of executed transactions happening in one second. Prices have been normalized and the average has been weighted by the number of shares traded in each transaction.
size - The sum number of shares traded.
order_count - The number of unique trade orders taking place.

### train.csv 

The ground truth values for the training set.

stock_id - Same as above, but since this is a csv the column will load as an integer instead of categorical.
time_id - Same as above.
target - The realized volatility computed over the 10 minute window following the feature data under the same stock/time_id. There is no overlap between feature and target data. You can find more info in our tutorial notebook.

### test.csv 

Provides the mapping between the other data files and the submission file. As with other test files, most of the data is only available to your notebook upon submission with just the first few rows available for download.

stock_id - Same as above.
time_id - Same as above.
row_id - Unique identifier for the submission row. There is one row for each existing time ID/stock ID pair. Each time window is not necessarily containing every individual stock.
sample_submission.csv - A sample submission file in the correct format.

row_id - Same as in test.csv.
target - Same definition as in train.csv. The benchmark is using the median target value from train.csv.

# Project implement

## step1: store data into AWS S3
a. go to AW3 S3 console, click "add bucket"
<img width="1120" alt="6b67a5e08b2cdc5960de721e65aeffc" src="https://user-images.githubusercontent.com/122952572/227016699-04d120c9-862e-4523-9cbe-d3d3c0336a2b.png">
b. upload your file into your bucket
<img width="764" alt="image" src="https://user-images.githubusercontent.com/123079408/235381520-7f52d5f5-695a-46d0-ac38-380b78edf955.png">

<img width="840" alt="WechatIMG219的副本" src="https://user-images.githubusercontent.com/123079408/235381580-53ffb3b3-b2ba-4753-b080-78cab901dcfe.png">


## step2: go to AWS sagemarket and lauch your notebook
a. go to AWS sagemarket console, click "notebook"
b. click "create notebook instance" 
<img width="1075" alt="3d5ca59a60f0f7fa354b150da58434e" src="https://user-images.githubusercontent.com/122952572/227017059-88c77066-1d13-4062-8bcb-37f4ab6be17f.png">
configure your notebook as following
![image](https://user-images.githubusercontent.com/122952572/227017476-4579bc12-2784-4c91-8998-c51755419723.png)

<img width="328" alt="b7af7c68ca98e648da88258a81b8fcd" src="https://user-images.githubusercontent.com/122952572/227017340-2717ed3e-7493-4b71-bfb7-8d30dde58c0f.png">

![image](https://user-images.githubusercontent.com/122952572/227017659-e551c0c9-ffd8-4958-921f-54cbfa79ead9.png)

c. wait a few minutes. In the Notebook instances section, the new SageMaker-Tutorial notebook instance is displayed with a Status of Pending. The notebook is ready when the Status changes to InService.
<img width="1120" alt="WechatIMG40" src="https://user-images.githubusercontent.com/123079408/235381659-e462d79d-9ab5-4513-b094-59f4d175d94e.png">

<img width="563" alt="09adc7afdd89deca12a112b1f6d7154" src="https://user-images.githubusercontent.com/122952572/227017905-cf1dabb2-d373-48fa-af5e-0a9a7ed3afbc.png">


d. After your SageMaker-Tutorial notebook instance status changes to InService, choose Open Jupyter.
e. In Jupyter, choose New and then choose conda_python3.

## step3: Amazon API Gateway deployment
API Gateway can present an external-facing, single point of entry for Amazon SageMaker endpoints:
<img width="1120" alt="image" src="https://user-images.githubusercontent.com/123079408/235381066-24421517-e392-4536-a81d-94e7b4078b89.png">

Then, use API to configure endpoint:
<img width="1100" alt="image" src="https://user-images.githubusercontent.com/123079408/235380910-4754558e-a6a1-4e4b-971a-05624fba180d.png">




## step4: load test - locust
We used locust for load teat
a. Install: pip3 install locust

b. Run: locust -f locustfile.py --host=https://7iz1uma2gk.execute-api.us-east-1.amazonaws.com

c. Once you have run it. Go to http://localhost:8089 in your browser and enter Locust's web console like this:

<img width="1280" alt="image" src="https://user-images.githubusercontent.com/123079408/235381186-cb3bc537-8112-4245-b18f-07bd5a013294.png">


Please check load test result:
<img width="858" alt="image" src="https://user-images.githubusercontent.com/123079408/235381233-3fb24c2e-07b6-432a-beba-2e18a746cc40.png">


# Overall Workflow
1. Prepare data (but need to go throght data exploration, data cleaning)
2. Train the ML model
3. Deploy the model
4. Load Test
5. Evaluate model performance
6. Clean up (Not terminating your resources will result in charges to your account.)
 
 
 For the full result, please refer to this https://github.com/ht175/IDS721-Project3/blob/main/start_up_prediction.ipynb
