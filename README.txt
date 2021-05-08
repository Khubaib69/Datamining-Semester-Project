		Software engineering team assessment and prediction(SETAP)


HOW TO RUN:
	Just change the file name to setapProcessT1.csv and save it and run it.

1- DataSet Introduction:
	
	in this dataset we have different groups of student that are enrolled in Software Engeineering subject and at the end of semester we have to predict
	weather they are pass or fail. we have dataset with 345 rows and 85 columns. columns include various parameters like number of students in group,
	number of coding hours,number of meeting attended by group etc. and we have to calculate A or F grade based on this information.

2- DataCleaning:
	
	The Main Problum with the dataset was that it was extremely baised towards the students with A grade and it is not good to train our model from a baised
	dataset so we take equal rows of A grade(120) students and F grade students(120). so now our dataset contain 120 rows with no baised data.

	luckily we have no rows with null values so it was good.

3- Test/Train Split:
	
	the last column of dataset which contain the actual grade of weather group of student are pass or fail are the independent variable and everyother column
	is dependent variable. since our dataset contain only 120 rows so i split it with 90% train and 10% test. so 216 for training and 24 for testing.

4- Applying ALgorithm:
	
	The first Algorithm that we applied is Logestic Regression we attain the accuracy of 83%. [[12  1]
 												  [ 3  8]]

	The second Algorithm that we applied is Naive Bayes we attain the accuracy of 66%. 

	The third Algorithm that we applied is  Random Forest Classifier we attain the accuracy of 58%. 

	The fourth Algorithm that we applied is k-nearest neighbors we attain the accuracy of 62%.


5- Testing and validation:
	
	we took one instance from database and saw the result of that instance. after that we passed that instance to our model trained on Logestic Regression
	(the best accuracy one) and let it predict. it predicted same result.