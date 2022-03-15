
## Project Name: Finding exoplanets using Star Flux Data from NASA
**Group Members: Samantha Wong, Megha Velakacharla, Lumyah Habib**


  <img src=https://user-images.githubusercontent.com/97067377/158479416-0326f0d0-ce97-4ef2-9953-b000126180b0.jpg width=1000/>



**Description:**

This project is about finding exoplanets in deep space using recorded changes in flux
or light intensity for 5087 stars. To do so, we will implement binary classification to classify stars with and without orbiting exoplanets. The input used to train our machine learning algorithm is a csv data set from Kaggle that documents 5087 stars, their binary classification, and 9 different time periods where their flux was recorded. The other input is the test data set containing another 570 stars with their recorded flux values, but without a classification. Our class contains a method that will analyze this test set of 570 stars and find the stars that have orbiting exoplanets using a decision tree and a k-nearest neighbor algorithm. Our class also contains a method that will compare the scores of different depths of the decision tree and number of neighbors considered in the KNN algorithm to find the best parameters for the models. Ultimately, the output will be the test data set with the addition of their binary classification.
  
Data set link: https://www.kaggle.com/keplersmachines/kepler-labelled-time-series-data
This is a public dataset from Kaggle containing cleaned data from observations made by the NASA Kepler space telescope. There are 68 notebooks of code, using different methods of machine learning to analyze the data.

**Package Installation:**
Packages are listed at top of py file, copy and paste them to run.

**Scope and Limitations:**
The current data uploaded using the csv file contains 3197 recordings of light intensities for each star. Because of the sheer amount of obsverations, it would take a long time for a new star to be added to the data. Currently, our class depends on every star having the same number of observations, so a future expansion to our class could add a parameter that changes the number of observations considered. Another potential extension is to add a method that visualizes the Knn model or the differences between predictions of the two models.

**Instructions:**
1) Load data file containing star light data. Each star should have 3197 instances where its light intensity is recorded. 
2) Load the train data csv file as an instance of the star_data class.
3) Run plot_tree_scores (depths = ), with depths being a numpy array of different depths to help find the best depth for the decision tree model. Default is 1 through 10.
  
      <img src=https://user-images.githubusercontent.com/97067377/158477906-da56888e-a922-4074-a734-6729ab98e5f4.jpg width="400"/>
  
5) Run plot_knn_scores ( n_neighbors = ), with n_neighbors being a numpy array of different number of neighbors to consider to find the best number of neighbors for the K-nearest neighbors model. Default is 1 through 20.

      <img src=https://user-images.githubusercontent.com/97067377/158477974-2ba6769f-d50f-4c4c-ae62-8e741b19f96a.jpg width="400"/>
 

7) Using the graph of scores, choose the best depth and number of neighbors for your decision tree and knn model.
8) Load or update a new instance of star_data with your train data csv file and specify the parameters d and n, with the best depth and number of neighbors respectively. 
9) Running the method plot_tree() from the star_data class will automatically split your train data into further train and faux test data, train the decision tree model, and visualize the branches.
   
   
   <img src=https://user-images.githubusercontent.com/97067377/158476277-bd47642b-d46a-42bd-b9b1-fc692c24a4e8.jpg width="400"/>
    

9) Running the method fit_knn() from the star_data class will automatically split your train data into further train and faux test data and then train the knn algorithm.
10) Use the all_tree_scores() and all_knn_scores() method in star_data class to print the accuracy scores of your models againist the faux test data in order to check for overfitting. 
11) Load your test data csv into a pandas dataframe using pd.read_csv('x').
12) Split your test data into predictor variables and outcome variable using **exoTest.drop(['LABEL'], axis=1)** for the predictor variables and **exoTest['LABEL']** for your outcome variable.
13) Now insert the predictions your models made about the test data into the test data dataframe using 
  **exoTest.insert(1, "Exoplanet Prediction by Tree", exo_best.fit_tree().predict(X1))**
  **exoTest.insert(2, "Exoplanet Prediction by Knn", exo_best.fit_knn().predict(X1))**
13) Now return the test data dataframe to view the predictions. A 1 in either of the prediction columns means an exostar is unconfirmed, while a 2 is a confirmation of an exostar.
    

![exoDF](https://user-images.githubusercontent.com/97067377/158475988-40ada8d3-8208-45ea-8e0e-b6ca7a983476.jpg)




