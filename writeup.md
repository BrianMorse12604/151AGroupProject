# 151AGroupProject

## A. Abstract

Airbnb is one of the largest short-term rental booking sites and a rental’s ratings can be instrumental in determining its profitability. As such, many hosts would like to know how a potential rental location may perform before they purchase it or how they can make improvements to current locations. Our goal is to create a predictive model that takes into account information about a rental location to predict the overall rating. We will use a regression model with features such as the number of rooms, the price of the listing, and the city of the listing that will predict the overall rating for the location. We will further process and transform the data to create new features in hopes of strengthening our predictions. We will explore various types of regression models including but not limited to linear regression, ridge regression, decision tree / random forests, and neural networks to determine which type of model best fits and predicts the data. By doing so, we hope to create a useful tool for hosts to use when making decisions about a listing.

Overview of Data: 
Data can be found on [kaggle](https://www.kaggle.com/datasets/mysarahmadbhat/airbnb-listings-reviews). We are using the 'Listings.csv' data and the corresponding 'Listings_data_dictionary.csv', which is just a dictionary describing all the fields in Listings.csv. 

This dataset uses a public domain license described here: [CC0: Public Domain](https://creativecommons.org/publicdomain/zero/1.0/). Based on this license, we are free to copy, modify, distribute and perform the work without asking permission. 

Objective: We are building a model to predict an Airbnb's review rating. 

## C. Methods

### Data Exploration and Initial Preprocessing (Milestone 2)

The initial data preprocessing and exploration is done in [project.ipynb](https://github.com/BrianMorse12604/151AGroupProject/blob/main/project.ipynb), linked in this github.

#### Data Preprocessing Steps:
The raw data has 279712 rows and 33 fields (columns), with 12 of these fields containing missing values. 

1. After looking through each field and its description, we dropped every column that could not be used to predict the review scores (target feature is 'review_scores_rating' which is an overall review rating): 'review_scores_accuracy', 'review_scores_cleanliness', 'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value'. We also dropped irrelevant columns: 'listing_id', 'name', 'host_id'.
2. Many of the host columns had the same number of missing values, and we confirmed that this meant the Airbnb didn't have information on those hosts (as many of the rows had missing values for all the columns 'host_since', 'host_is_superhost', 'host_total_listings_count', "host_has_profile_pic", "host_identity_verified"). We deleted all the rows in which the Airbnb did not have information on the host, as well as the columns that had too many missing values: host response times, host response rates, district, 
3. Many rows had missing values for the number of bedrooms, and it was a tough decision but ultimately we opted to drop all rows where the number of bedrooms was missing, considering that it'd be tough to make a decision on booking an AirBNB without its number of bedrooms.
4. We decided latitude and longitude would be redundant when city is already provided, so we dropped that variable. We'll also drop neighborhood, as we feel it's unlikely to have enough data points for each neighborhood to be of use. We'll also drop host_location, as we don't see it being effective in predicting rating.
5. Finally, we're also going to have to drop all rows where review_scores_rating is null, as this is the variable we need to predict.

To ensure only numerical data: All the columns that have t/f, we converted to binary 1/0. All columns with categorical variables, with the exception of amenities, we one-hot encoded. This is "property_type", "room_type", "city". We only included the top 100 amenities. 

#### Normalizing the Data
From a boxplot of the nonbinary features, we found that features such as 'host_total_listings_count,' 'price,' 'minimum_nights,' and 'maximum_nights' have extreme outliers that we can drop. We will drop the first 10 outliers of the previously notable features mentioned. We have opted not to employ the IQR method since it would excessively constrain our dataset.

After dropping these outliers, we normalized the data.

#### Data Visualization
To get a better sense of the data:

1. We first plotted the correlations between different features, specifically the non-categorical fields. We found that the strongest correlation between our intended target class, review_scores_rating, was host_is_superhost. The next 9 features with the strongest correlation to ratings are a variety of amenities relating to food and cooking: Dishes and silverware, Cooking basics, Coffee maker, Refrigerator, Stove, Oven, Hot water, Iron, and Microwave.

2. We looked at how pricing could potentially affect rating, since we preliminary hypothesized that pricing could be a major indicator of rating. From the scatterplot, we found that there does appear to be less points at the higher end of 'price' and at the lower end of the review rating. We broke this down even more to view price plotted against review by each city. These plots showed us the same pattern, with higher priced Airbnbs having fewer lower ratings.

3. Since being a superhost was found to be our strongest correlating factor, we plotted the distribution of ratings based on superhost against the rating. We found that if the host is a superhost, the ratings are much more skewed to have a higher concentration among the 90-100% rating. Similarly, if the host is not a super host it is seen that they have slightly more reviews amongst the 20-50% range.

### First Model - Regression (Milestone 3) 
This is the link to the our notebook for our [regression model](https://github.com/BrianMorse12604/151AGroupProject/blob/main/regression.ipynb) linked in this github. 

#### Preprocessing for our Model  
In this model, the only extra preprocessing we did to add to the work done from the previous milestone was to incorporate polynomial features up to degree three and interactions for all of the features that were not binary. This was done to experiment with the features that may be important beyond simple linearity without causing too many issues since the recursive feature elimination would be able to remove any added columns that were problematic.

We did not add this preprocessing directly to the data that other models would use and instead only saved this preprocessing for our regression in case the other models would not benefit from it. However, the code is written in a function such that it could easily be incorporated for future models if desired.

#### Evolution of the Model

The first step in doing regression was to use do a simple linear regression with no extra bells or whistles attached. After that, a ridge regression model was created to explore the differences that that would create. Then a second ridge regression model was made that included the polynomial data and had recursive feature elimination cross validation to only keep the best features for the regression calculations. After some messing around with the hyperparameters, we had the RFE have ten splits and remove ten features at each step, though changing these parameters had minimal effect.

### Second Model - Decision Tree Regressor (Milestone 4) 

This is the link to the our notebook for our [tree models](https://github.com/BrianMorse12604/151AGroupProject/blob/main/trees.ipynb) linked in this github. 

#### Part 1: Evaluating data, labels, and loss function

We felt that our data, labels, and loss functions were all sufficient. We knew that regression models are more simple models that often struggle with identifying nonlinear relationships, so we were pleasantly surprised to see the MSE hover around 90 – suggesting an average error around 9-10 / 100. Our data is extensive both in terms of observations and features, so we were excited to get into more complex models that could model relationships that could be harder to notice.

#### Part 2: Train your second model

We began with decision tree regressors and random forest regressors. We then used both XGBoost and scikit-learn’s libraries for gradient boosting regressors. Fitting each model was relatively quick, finishing in ~10 minutes.

#### Part 5: Hyperparameter tuning, K-fold cross-validation, feature expansion

We did not perform feature expansion, as we felt confident that our feature engineering in the preprocessing stage was effective. We did perform both hyperparameter tuning and K-Fold cross validation using the GridSearchCV function. Though the grid search did find slightly more optimal errors in comparison to our models without hyperparameter tuning, we weren’t blown away by the results. 
We ran trials using a different number of folds during our grid search, and found that the difference in results was only marginal, while each search took an order of magnitude longer. Therefore, we opted to gridsearch with only two folds each time we used it, using the second fold to sanity-check the results of the first. 

### Third Model - Dense Neural Networks (Milestone 5)


## D. Results

### Data Exploration and Initial Preprocessing (Milestone 2)

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

### First Model - Regression (Milestone 3) 

#### Observation 
Our first initial model, which consisted of a simple Linear Regression model, produced an extremely high validation error of 1.2e+18 in comparison with its training error of 90.6 (around 10^16 times greater). The Ridge Regression model produced very similar errors between the validation and training dataset of roughly 88.1 and 90.6 respectively. Although much better than the Linear Regression model, we made one more attempt at improving the model by using feature transformation to consider polynomial features and feature removal to remove unnecessary and non-impactful features. However, this ultimately did not improve the model as we had hoped, and our new reduced Ridge Regression model ended up producing very similar errors as the previous Ridge Regression model, 87.7 and 90.1 respectively. Following the training and finalization of our model, we tested the reduced Ridge Regression model on the testing dataset and it produced an error of 90.6.

#### Fitting Graph 
Our final reduced Ridge Regression model produced validation, training, and testing errors of roughly 87.7, 90.1, and 90.6 respectively. We can observe that all of the errors are relatively close together, leading us to conclude that we are no longer overfitting, but may be underfitting the data due to the high error values. 

![Fitting Curve for Regression](images/regression_error_plotting.png)

The error plotting has to be in log to try to plot all the points in the graph, but it is clear that after the first drop in validation error that none of the error significantly changed for the better or worse, but made very marginal improvement, confirming the analysis above. The model number label on the x-axis essentially describes the separation of the three major sections of progress with regression, model 0 being the standard linear regression, model 1 being the ridge regression, and model 2 being the regression that has some polynomial features and recursive feature elimination applied to it.

### Second Model - Decision Tree Regressor (Milestone 4) 

#### Part 3: Evaluate the Model

We decided to use the gradient boosting regression model from the XGBoost library, as it performed slightly better than both the decision tree regressor and the random forest regressor.
Our best gradient boosting model overfits slightly, with a training MSE of 68 and testing/validation MSE’s hovering in the mid-80’. However, as the result of an exhaustive grid search, we have confidence that these hyperparameters are pretty optimal, as they outperformed models that didn’t overfit and had similar training and testing errors. l Tweaking the “num_estimators” and “depth” hyperparameters had the largest impact on mitigating overfitting: we were able to bring the training MSE down to a single digit, at the cost of doubling our training/validation MSE’s by building an overly complex model with high depth.

#### Part 4: Where does the model fit in the fitting graph?

![Fitting Curve for Regression](images/XGBoost_plotting.png)

Our XGBoost model had training, testing, and validation errors were roughly 68.5, 86.3, and 84.8 respectively. Looking at our XGBoost RMSE graph, our model overfits slightly as the final training and testing RMSE has a decent difference. However, the error on both our test and validation data is not yet in the area of the fitting graph where it is starting to rise as the training RMSE continues to fall. Therefore, we can conclude that our model falls roughly in the ideal range for model complexity.

### Third Model - Dense Neural Networks (Milestone 5)


## E. Discussion

### Data Exploration and Initial Preprocessing (Milestone 2)


### First Model - Regression (Milestone 3) 

#### Results Analysis

The difference in validation and training error in combination with extreme coefficient values was a clear sign of overfitting for the original linear regression model and a failure on the model’s part to understand the true importance of different features. We attempted to fix this issue by implementing a Ridge Regression to counter larger coefficients and reduce the error, which did in fact make a significant difference. However, when trying to make further improvement with polynomial data and feature elimination, there was not as much improvement as expected, perhaps since a lot of features do not actually make a huge impact on a user and thus would not effect the model whether the feature is included or not.

In terms of where we are on the fitting graph with this model, we are on the left side before the ideal range for model complexity, with a simple model and high predictive errors. We also noticed that the validation error never seemed to increase at all as we increased the model complexity, further indicating that we were before the ideal range for model complexity.

#### Conclusion and Next Steps 
Upon exploring a Linear Regression model, we came to the conclusion that Linear Regression may not be sufficient for predicting Airbnb's review rating given our dataset. This is demonstrated in both our initial Linear Regression model and our improved reduced Ridge Regression Model. Although consistent, our final model still produced high error values, indicating it may not be accurately predicting the rating reviews as much as a model could be. 

One possible area of improvement for this regression model includes further investigation into the importance of each of the features and their relationships with each other to see what the best subset of features truly is to limit the error, but currently, it seems like most changes would not be significant for this model’s improvement and that it would most likely only make a small decrease.

The next two models we are thinking of doing are decision regression trees and neural networks. This is because those are two other common types of models that tend to do well with regression problems and may be able to overcome the challenges that a regression model can not. These more complicated models have automatic feature learning and can recognize nonlinear relationships between data. Therefore, compared to a traditional regression model that performs well with linear data relationships, our future models may be able to recognize underlying patterns and fit the data more, thus decreasing the final error.

### Second Model - Decision Tree Regressor (Milestone 4) 

#### Part 6: Plan for the next model

For our next model, we were planning to test various types of neural networks. We saw that several neural networks performed well on similar tasks in our recent homework, so we felt that it would be interesting to see if that translated to this task as well. We plan on starting out with a simple Multi-layer perceptron, then experimenting with ANN’s and DNN’s with varying depths, activation functions, and number of neurons.

#### Part 7: Model 2 Conclusion

Our second model only improved marginally upon our first model. This family of models – the trees – had a tendency to overfit on the training data, which made sense given the complexity of each of these models. We think that these different models performing relatively similar to one another is indicative of the variance in our dataset; people, and therefore AirBNB ratings, are inherently volatile. We think that there’s a possibility that more hyperparameter tuning can decrease the error slightly, but we’re not overly hopeful that this is the case given the similar performance across the board.

### Third Model - Dense Neural Networks (Milestone 5)



## F. Conclusion

After the entire process of working with this Airbnb data across three different models, we were able to successfully predict the ratings of different locations within 9-10%. That means we are able to typically predict with an error of half of a star which is pretty good. Surprisingly, it seems that the tree models were the best performing in completing this task, even above the neural networks. However, this still clearly demonstrates that there is still a lot of room for improvement to get even more accurate results. Given more time with the models we were working on, we may be able to implement more tactics to have better predictive power or do more hyperparameter tuning to have the best architecture available for the respective model types. One more thing to consider however is that since all three models tended to have the same loss as each other, it may also be more beneficial to have a deeper and more complex look into the data and try to find how some of the features truly effect the outcome and each other. There may also be bias into how the data was collected in the first place, like how the concentration of scores tend to be distributed and be heavily skewed positively, in which case we would need to look for other data to either supplement or replace the current data we were looking at. However, despite the different paths forward from here, we have still created a good model to be used for those looking to start putting a place up on Airbnb or for users to get an idea of what their experience may be like if there are not many reviews on the location yet!
