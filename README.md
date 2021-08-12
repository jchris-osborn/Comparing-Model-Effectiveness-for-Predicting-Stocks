# Comparing-Model-Effectiveness-for-Predicting-Stocks
## ABSTRACT 

The stock market is a fickle beast. For years economists have been trying to tame the beast. Predictive models have become one tool for the economist. Due to the uncontrollable nature of the stock market, making predictions is like finding a needle in a haystack. However, people and economists alike still search for the answer. One of the challenges in uses predictive models is finding the right model for the data type or even the data set. Many models exist; it is easy to get lost before getting started. This experiment attempts to find models that are efficient and effective at predicting future prices of stocks. The experiment consists of ten stocks being predicted by five different models: SVR, LSVR, kNN, Random Forest, and Bagging. The stock data sets are from Kaggle.com and contain the daily life of the stock. Each data set includes the Date, opening price, daily high price, daily low price, closing price, and the volume of the stock traded that day. The target value for modeling is the closing price: Close. The stocks were chosen at random to help reduce the bias in the experiment. The data sets were complete, meaning there was no missingness or NAN. The only pre-processing done was to the volume—the scale and the overall skewness. The skewness and the volume were taken care of by a single log transformation. The volume values went from in the millions to the teens. An added benefit of the log transformation was that it reduced the overall number of outliers. Remaining outliers were kept in the data set because they are believed to be a good indication of the overall trend of the stock. The data sets were then split into training and test set. Once the data was trained and tested, I used RMSE, MAE, and R2 to judge the model’s performance. All the models performed relatively well; random forest performed the best. Both its errors were low, and the R2 was above .9. The results in this experiment suggest using random forest when forecasting stock prices. 

## DATA 

The data for this experiment was obtained from Kaggle.com. Kaggle is a data science website that holds data science competitions. They also have thousands of reliable data sets. The data set chosen for this experiment is the “Huge Stock Market Dataset.” It is a comprehensive historical daily price and volume data for all US-based stocks. It contains the life of each stock up until 2017. The data was last updated on 11/10/2017. Each stock includes the attributes Date, Open, High, Low, Close, Volume, OpenInt. Open, high, low, and close are all prices of the stock throughout the day. Volume is the amount that it was traded during the day. The target variable is Close, the daily closing price of the stock.  Below is an example of a data set. 

## RESULTS

All the models performed relatively well with low error and high R2. Both of these measures indicate that the models were able to fit the data pretty well.  While all the models performed well, there were some discrepancies between the models. Overall the SVM models, SVR and LSVR, performed the poorest. Their errors were generally higher for every stock. kNN performed much better than the SVM models but not quite as good as Bagging and Random Forest. The model that performed the best was random forest. Random forest had across all stocks had an average RME of .3502, an average MAE of 0.21407, and a mean R2 of 0.99977. Random forest outperformed every model, for every stock, in RSME, MAE, and r-squared. 
