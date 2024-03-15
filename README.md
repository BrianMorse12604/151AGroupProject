# Introduction to 151AGroupProject

The group members in this project are Rachel Wei (rawei@ucsd.edu), Andrew Pu (apu@ucsd.edu), Ethan Cao (etcao@ucsd.edu), idk everyone elses email im too lazy. 

This README will explain our 151A group project for Winter 2024. This README contains where to find all the code for this project, as well as the project itself: an introduction to the project, dataset used, a description of our data exploration, cleaning, and preprocessing, and the process for creating the 3 different ML models as well as a comparison between the 3. All the code for this project is uploaded as a jupyter notebook to this github, and will be linked throughout the readme when relevant. 

# Project Writeup 

## Introduction to Project (Project Abstract) 

Airbnb is one of the largest short-term rental booking sites and a rental’s ratings can be instrumental in determining its profitability. As such, many hosts would like to know how a potential rental location may perform before they purchase it or how they can make improvements to current locations. Our goal is to create a predictive model that takes into account information about a rental location to predict the overall rating. We will use a regression model with features such as the number of rooms, the price of the listing, and the city of the listing that will predict the overall rating for the location. We will further process and transform the data to create new features in hopes of strengthening our predictions. 

We will explore 3 ML models: 
1. Various linear regression mdoels, in which we pick the model that performed the best.
2. Various tree-based ML models, in which we pick the model that performed the best.
3. Various DNN models, in which we pick the model that performed the best.

By creating these models, we hope to create a useful tool for both hosts and customers to use when making decisions about a listing.

Objective: We are building 3 ML models to predict an Airbnb's review rating.   

## Figures

The following figures are for data visualization purposes, and to get a better sense of the data

[Figure 1.1](https://github.com/BrianMorse12604/151AGroupProject/edit/main/README.md#data-visualization): This figure is described more thoroughly at the link, but gives the correlation between different features of the data.
![151a_correlation_dataVis](https://github.com/BrianMorse12604/151AGroupProject/assets/40574565/e985ecb3-a7f3-4d16-b394-caac5080e3e8)

[Figure 1.2](https://github.com/BrianMorse12604/151AGroupProject/edit/main/README.md#data-visualization): This figure is described more thoroughly at the link, but compares price against rating for each location.
![151a_pricevshost_dataVis](https://github.com/BrianMorse12604/151AGroupProject/assets/40574565/e1840db3-7314-4cca-8765-731cb08437ce)

[Figure 1.3]([url](https://github.com/BrianMorse12604/151AGroupProject/edit/main/README.md#data-visualization)): This figure is described more thoroughly at the link, but looks at the distribution of ratings based on whether the host is a host or superhost.

![151a_superhost_dataVis](https://github.com/BrianMorse12604/151AGroupProject/assets/40574565/69629b7b-9401-4762-8e7c-81d5118ccc1f)

## Data Exploration and Initial Preprocessing (Milestone 2)

Overview of Data: 
The dataset we used can be found on [kaggle](https://www.kaggle.com/datasets/mysarahmadbhat/airbnb-listings-reviews). We are using the 'Listings.csv' data and the corresponding 'Listings_data_dictionary.csv', which is just a dictionary describing all the fields in Listings.csv. 

This dataset uses a public domain license described here: [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/). Based on this license, we are free to copy, modify, distribute and perform the work without asking permission. 

The initial data preprocessing and exploration is done in [project.ipynb](https://github.com/BrianMorse12604/151AGroupProject/blob/main/project.ipynb), linked in this github.

#### Data Exploration and Preprocessing:
The raw data has 279712 rows and 33 fields (columns), with 12 of these fields containing missing values. 

1. After looking through each field and its description, we dropped every column that could not be used to predict the review scores (target feature is 'review_scores_rating' which is an overall review rating): 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value'. We also dropped irrelevant columns: 'listing_id', 'name', 'host_id'.
2. Many of the host columns had the same number of missing values, and we confirmed that this meant the Airbnb didn't have information on those hosts (as many of the rows had missing values for all the columns 'host_since', 'host_is_superhost', 'host_total_listings_count', "host_has_profile_pic", "host_identity_verified"). We deleted all the rows in which the Airbnb did not have information on the host, as well as the columns that had too many missing values: host response times, host response rates, district, 
3. Many rows had missing values for the number of bedrooms, and it was a tough decision but ultimately we opted to drop all rows where the number of bedrooms was missing, considering that it'd be tough to make a decision on booking an AirBNB without its number of bedrooms.
4. We decided latitude and longitude would be redundant when city is already provided, so we dropped that variable. We'll also drop neighborhood, as we feel it's unlikely to have enough data points for each neighborhood to be of use. We'll also drop host_location, as we don't see it being effective in predicting rating.
5. Finally, we're also going to have to drop all rows where review_scores_rating is null, as this is the variable we need to predict.

After this process, our data has 168414 entries with 16 fields. This preserved 60% of the observations. The following is a breakdown of the key columns we chose to keep, and the rational behind keeping them: 

- host_since -- we want the amount of time a host has been active, as we think this would've given them time to adjust their properties to be of higher quality. We converted this into number of days the host has been active.
- All other host variables -- gives us a decent idea of the quality of the host.
- city -- gives us a good idea of their rough location.
- property_type and room_type -- the type of housing may affect how well they're generally rated
- accomodates / bedrooms -- the size of the AirBNB might follow general trends in rating
- amenities -- gives us the added benefits, which may add a convenience that'll boost rating.
- price -- rating may be correlated with pricing, one way or another. The price is initially listed in the Airbnb's local price, which we converted to USD to be consistent across the entire data.
- minimum_nights and maximum_nights -- we think that this'll add to or subtract from the convenience for the user, leading to correlation with rating
- instant_bookable -- same as above.

To ensure only numerical data: All the columns that have t/f, we converted to binary 1/0. All columns with categorical variables, with the exception of amenities, we one-hot encoded. This is "property_type", "room_type", "city". We only included the top 100 amenities. 

#### Normalizing the Data
From a boxplot of the nonbinary features, we found that features such as 'host_total_listings_count,' 'price,' 'minimum_nights,' and 'maximum_nights' have extreme outliers that we can drop. We will drop the first 10 outliers of the previously notable features mentioned. We have opted not to employ the IQR method since it would excessively constrain our dataset.

After dropping these outliers, we normalized the data.

#### Data Visualization
To get a better sense of the data:

1. [Figure 1.1](https://github.com/BrianMorse12604/151AGroupProject/edit/main/README.md#figures): We first plotted the correlations between different features, specifically the non-categorical fields. We found that the strongest correlation between our intended target class, review_scores_rating, was host_is_superhost. The next 9 features with the strongest correlation to ratings are a variety of amenities relating to food and cooking: Dishes and silverware, Cooking basics, Coffee maker, Refrigerator, Stove, Oven, Hot water, Iron, and Microwave.

2. [Figure 1.2](https://github.com/BrianMorse12604/151AGroupProject/edit/main/README.md#figures): We looked at how pricing could potentially affect rating, since we preliminary hypothesized that pricing could be a major indicator of rating. From the scatterplot, we found that there does appear to be less points at the higher end of 'price' and at the lower end of the review rating. We broke this down even more to view price plotted against review by each city. These plots showed us the same pattern, with higher priced Airbnbs having fewer lower ratings.

3. [Figure 1.3](https://github.com/BrianMorse12604/151AGroupProject/edit/main/README.md#figures): Since being a superhost was found to be our strongest correlating factor, we plotted the distribution of ratings based on superhost against the rating. We found that if the host is a superhost, the ratings are much more skewed to have a higher concentration among the 90-100% rating. Similarly, if the host is not a super host it is seen that they have slightly more reviews amongst the 20-50% range.

## First Model - Regression (Milestone 3) 
This is the link to the our notebook for our [regression model](https://github.com/BrianMorse12604/151AGroupProject/blob/main/regression.ipynb) linked in this github. 


#### Preprocessing for our Model  
In this model, the only extra preprocessing we did to add to the work done from the previous milestone was to incorporate polynomial features up to degree three and interactions for all of the features that were not binary. This was done to experiment with the features that may be important beyond simple linearity without causing too many issues since the recursive feature elimination would be able to remove any added columns that were problematic.

We did not add this preprocessing directly to the data that other models would use and instead only saved this preprocessing for our regression in case the other models would not benefit from it. However, the code is written in a function such that it could easily be incorporated for future models if desired.

#### Observation 
Our first initial model, which consisted of a simple Linear Regression model, produced an extremely high validation error of 1.2e+18 in comparison with its training error of 90.6 (around 10^16 times greater). The difference in validation and training error in combination with extreme coefficient values was a clear sign of overfitting and a failure on the model’s part to understand the true importance of different features. We attempted to fix this issue by implementing a Ridge Regression to counter larger coefficients and reduce the error. The Ridge Regression model produced very similar errors between the validation and training dataset of roughly 88.1 and 90.6 respectively. Although much better than the Linear Regression model, we made one more attempt at improving the model by using feature transformation to consider polynomial features and feature removal to remove unnecessary and non-impactful features. However, this ultimately did not improve the model as we had hoped, and our new reduced Ridge Regression model ended up producing very similar errors as the previous Ridge Regression model, 87.7 and 90.1 respectively. Following the training and finalization of our model, we tested the reduced Ridge Regression model on the testing dataset and it produced an error of 90.6.

#### Fitting Graph 
Our final reduced Ridge Regression model produced validation, training, and testing errors of roughly 87.7, 90.1, and 90.6 respectively. We can observe that all of the errors are relatively close together, leading us to conclude that we are no longer overfitting, but may be underfitting the data due to the high error values. In terms of where we are on the graph, we are on the left side before the ideal range for model complexity, with a simple model and high predictive errors. We also noticed that the validation error never seemed to increase at all as we increased the model complexity, further indicating that we were before the ideal range for model complexity.

![Fitting Curve for Regression](images/regression_error_plotting.png)

The error plotting has to be in log to try to plot all the points in the graph, but it is clear that after the first drop in validation error that none of the error significantly changed for the better or worse, but made very marginal improvement, confirming the analysis above. The model number label on the x-axis essentially describes the separation of the three major sections of progress with regression, model 0 being the standard linear regression, model 1 being the ridge regression, and model 2 being the regression that has some polynomial features and recursive feature elimination applied to it.

#### Conclusion and Next Steps 
Upon exploring a Linear Regression model, we came to the conclusion that Linear Regression is not sufficient for predicting Airbnb's review rating given our dataset. This is demonstrated in both our initial Linear Regression model and our improved reduced Ridge Regression Model. Although consistent, our final model still produced high error values, indicating it was not accurately predicting the rating reviews. 


One possible area of improvement for this regression model includes further investigation into the importance of each of the features and their relationships with each other to see what the best subset of features truly is to limit the error, but currently, it seems like most changes would not be significant for this model’s improvement and that it would most likely only make a small decrease.


The next two models we are thinking of doing are neural networks and decision regression trees. This is because those are two other common types of models that tend to do well with regression problems and may be able to overcome the challenges that a regression model can not. These more complicated models have automatic feature learning and can recognize nonlinear relationships between data. Therefore, compared to a traditional regression model that performs well with linear data relationships, our future models may be able to recognize underlying patterns and fit the data more, thus decreasing the final error.

## Second Model - Decision Tree Regressor (Milestone 4) 

This is the link to the our notebook for our [tree models](https://github.com/BrianMorse12604/151AGroupProject/blob/main/trees.ipynb) linked in this github. 

### Part 1: Evaluating data, labels, and loss function

We felt that our data, labels, and loss functions were all sufficient. We knew that regression models are more simple models that often struggle with identifying nonlinear relationships, so we were pleasantly surprised to see the MSE hover around 90 – suggesting an average error around 9-10 / 100. Our data is extensive both in terms of observations and features, so we were excited to get into more complex models that could model relationships that could be harder to notice.

### Part 2: Train your second model

We began with decision tree regressors and random forest regressors. We then used both XGBoost and scikit-learn’s libraries for gradient boosting regressors. Fitting each model was relatively quick, finishing in ~10 minutes.

### Part 3: Evaluate the Model

We decided to use the gradient boosting regression model from the XGBoost library, as it performed slightly better than both the decision tree regressor and the random forest regressor.
Our best gradient boosting model overfits slightly, with a training MSE of 68 and testing/validation MSE’s hovering in the mid-80’. However, as the result of an exhaustive grid search, we have confidence that these hyperparameters are pretty optimal, as they outperformed models that didn’t overfit and had similar training and testing errors. l Tweaking the “num_estimators” and “depth” hyperparameters had the largest impact on mitigating overfitting: we were able to bring the training MSE down to a single digit, at the cost of doubling our training/validation MSE’s by building an overly complex model with high depth.

### Part 4: Where does the model fit in the fitting graph?

![Fitting Curve for Regression](images/XGBoost_plotting.png)

Our XGBoost model had training, testing, and validation errors were roughly 68.5, 86.3, and 84.8 respectively. Looking at our XGBoost RMSE graph, our model overfits slightly as the final training and testing RMSE has a decent difference. However, the error on both our test and validation data is not yet in the area of the fitting graph where it is starting to rise as the training RMSE continues to fall. Therefore, we can conclude that our model falls roughly in the ideal range for model complexity.

### Part 5: Hyperparameter tuning, K-fold cross-validation, feature expansion

We did not perform feature expansion, as we felt confident that our feature engineering in the preprocessing stage was effective. We did perform both hyperparameter tuning and K-Fold cross validation using the GridSearchCV function. Though the grid search did find slightly more optimal errors in comparison to our models without hyperparameter tuning, we weren’t blown away by the results. 
We ran trials using a different number of folds during our grid search, and found that the difference in results was only marginal, while each search took an order of magnitude longer. Therefore, we opted to gridsearch with only two folds each time we used it, using the second fold to sanity-check the results of the first. 

### Part 6: Plan for the next model

For our next model, we were planning to test various types of neural networks. We saw that several neural networks performed well on similar tasks in our recent homework, so we felt that it would be interesting to see if that translated to this task as well. We plan on starting out with a simple Multi-layer perceptron, then experimenting with ANN’s and DNN’s with varying depths, activation functions, and number of neurons.

### Part 7: Model 2 Conclusion

Our second model only improved marginally upon our first model. This family of models – the trees – had a tendency to overfit on the training data, which made sense given the complexity of each of these models. We think that these different models performing relatively similar to one another is indicative of the variance in our dataset; people, and therefore AirBNB ratings, are inherently volatile. We think that there’s a possibility that more hyperparameter tuning can decrease the error slightly, but we’re not overly hopeful that this is the case given the similar performance across the board.
