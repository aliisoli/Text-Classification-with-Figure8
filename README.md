# Disaster Response Text Classification Pipeline Project

## Summary:
This project is one of the deliverables for Udacity's Data Science Nanodegree program. In this project, we aim at categorizing text messages that are sent during a disaster and check if they are containing information that can save the lives of the people affected by the disaster.

### Files:
process_data.py contains an ETL pipeline that cleans the data from csv files and stores the cleaned data in sql databases
train_classifier.py contains a machine learning pipeline that does a grid search to optimie model parameters and stores the model in a pickle file for later use.

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
        
2.  In the terminal, use this command to get the link for vieweing the app:
env | grep WORK

The link wil be:
http://WORKSPACESPACEID-3001.WORKSPACEDOMAIN replacing WORKSPACEID and WORKSPACEDOMAIN with your values.


3. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Insert your own text into the search bar and see the classification results

