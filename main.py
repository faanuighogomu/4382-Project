# import packages
import pandas as pd
import numpy as np
# import seaborn as sns
import sklearn

# load the work from home data
wfh_data = pd.read_csv('4382 Project Data.csv')
print(wfh_data)

# the input variables inlcude all of the columns except for our target variable (mental health condition) and employee id
var = ['Age', 'Gender','Job_Role', 'Industry', 'Years_of_Experience', 'Work_Location', 'Hours_Worked_Per_Week', 'Number_of_Virtual_Meetings', 'Work_Life_Balance_Rating', 'Stress_Level', "Access_to_Mental_Health_Resources", 'Productivity_Change', 'Social_Isolation_Rating', 'Satisfaction_with_Remote_Work', 'Company_Support_for_Remote_Work', 'Physical_Activity', 'Sleep_Quality', 'Region']
# set all the input variables equal to variable X
X = wfh_data[var]

#output (dependent) variable, each record will have a value of either Anxiety, Burnout, Depression, or None
y = wfh_data['Mental_Health_Condition']

# split and train the data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=42)
