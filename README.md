# IDS721-Project3
# Introduction
This is the individual project3 for the course IDS721. I used XGBoost/Random forest/PCA in AWS SageMaker to train the model based on startup Dataset and tested the trained model using SageMaker's Batch Transform functionality. I also compares the actual and predicted values in a table called a confusion matrix. Besides, I implemented S3 to store the data in the cloud.

# Dataset Context
A startup or start-up is a company or project begun by an entrepreneur to seek, develop, and validate a scalable economic model. While entrepreneurship refers to all new businesses, including self-employment and businesses that never intend to become registered, startups refer to new businesses that intend to grow large beyond the solo founder. Startups face high uncertainty and have high rates of failure, but a minority of them do go on to be successful and influential. Some startups become unicorns: privately held startup companies valued at over US$1 billion. [Source of information: Wikipedia]
Startups play a major role in economic growth. They bring new ideas, spur innovation, create employment thereby moving the economy. There has been an exponential growth in startups over the past few years. Predicting the success of a startup allows investors to find companies that have the potential for rapid growth, thereby allowing them to be one step ahead of the competition.

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

# overall workflow
1. Prepare data (but need to go throght data exploration, data cleaning)
2. Train the ML model
3. Deploy the model
4. Evaluate model performance
5. Clean up (Not terminating your resources will result in charges to your account.)
 
 
 For the full result, please refer to this https://github.com/ht175/IDS721-Project3/blob/main/start_up_prediction.ipynb
