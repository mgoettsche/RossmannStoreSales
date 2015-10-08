# Slightly different features and NA handling than in first random forest approach.
# Model building using xgboost package and averaging of two models.
# Leaderboard score ~0.108.

### Libraries ###
library(lubridate)
library(xgboost)

### Globals ###
set.seed(321)


### Data ###
sample_submission_file <- "data/sample_submission.csv"
stores_file <- "data/store.csv"
test_file <- "data/test.csv"
train_file <- "data/train.csv"

sample_submission <- read.csv(sample_submission_file)
stores_raw <- read.csv(stores_file)
test_raw <- read.csv(test_file)
train_raw <- read.csv(train_file)


### Feature Engineering: Train/Test ###
# Expand Date to Year/Month/Day columns
train <- cbind(train_raw, Year = year(train_raw$Date), Month = month(train_raw$Date), Day = day(train_raw$Date))
test <- cbind(test_raw, Year = year(test_raw$Date), Month = month(test_raw$Date), Day = day(test_raw$Date))

# Remove un-expanded Date column
columnsToRemove <- c("Date")
train <- train[,!(names(train) %in% columnsToRemove)]
test <- test[,!(names(test) %in% columnsToRemove)]

# Convert StateHoliday to integer
train$StateHoliday <- as.integer(train$StateHoliday)
test$StateHoliday <- as.integer(test$StateHoliday)

# Re-order columns for visual inspection
# StateHoliday == (b|c) not present in test set, thus removing the columns.
# In real-word application you would want to keep them.
ordering_train <- c("Store", "Year", "Month", "Day", "DayOfWeek", "Sales", "Customers", "Open", "Promo",
              "SchoolHoliday", "StateHoliday")
ordering_test <- c("Id", "Store", "Year", "Month", "Day", "DayOfWeek", "Open", "Promo",
              "SchoolHoliday", "StateHoliday")
train <- train[,ordering_train]
test <- test[,ordering_test]


### Feature Engineering: Stores ###
# Convert StoreType to integer
stores <- stores_raw
stores$StoreType <- as.integer(stores$StoreType)

# Convert Assortment to integer
stores$Assortment <- as.integer(stores$Assortment)

# Convert PromoInterval to integer
stores$PromoInterval <- as.integer(stores$PromoInterval)

# Some 11 test data entries of store 622 have Open==NA. Assume Open==1, because there were
# no holidays.
test$Open[is.na(test$Open)] <- 1

# Promo2SinceWeek and Promo2SinceYear can hardly be imputed. Ignore them in model building
# for the moment.

### Join Train/Test and Stores ###
train <- merge(train, stores)
test <- merge(test, stores)

### Missing Data ###
# Only include days where store was open and sales not zero.
train <- train[ which(train$Open=='1'),]
train <- train[ which(train$Sales!='0'),]
# Replace NAs with 0
train[is.na(train)] <- 0
test[is.na(test)] <- 0


# Error metric
RMPSE<- function(preds, dtrain) {
  labels <- getinfo(dtrain, "label")
  elab<-exp(as.numeric(labels))-1
  epreds<-exp(as.numeric(preds))-1
  err <- sqrt(mean((epreds/elab-1)^2))
  return(list(metric = "RMPSE", value = err))
}


### Modelling ###
# Model 1
trainSales <- train$Sales
train <- train[,-c(6, 7)]
validation<-sample(nrow(train),10000)
dval<-xgb.DMatrix(data=data.matrix(train[validation,]),label=log(trainSales+1)[validation])
dtrain<-xgb.DMatrix(data=data.matrix(train[-validation,]),label=log(trainSales+1)[-validation])
watchlist<-list(val=dval,train=dtrain)

param <- list(objective = "reg:linear", booster = "gbtree",
                eta = 0.10, max_depth = 8, subsample = 0.7,
                colsample_bytree = 0.7)
model1 <- xgb.train(params = param, data = dtrain, nrounds = 1300, 
                    verbose = 1, early.stop.round = 30,
                    watchlist = watchlist, maximize = FALSE,
                    feval = RMPSE )

pred1 <- exp(predict(model1, data.matrix(test[,-c(2)])))-1


# Model 2
validation<-sample(nrow(train),10000)
dval<-xgb.DMatrix(data=data.matrix(train[validation,]),label=log(trainSales+1)[validation])
dtrain<-xgb.DMatrix(data=data.matrix(train[-validation,]),label=log(trainSales+1)[-validation])
watchlist<-list(val=dval,train=dtrain)

model2 <- xgb.train(params = param, data = dtrain, nrounds = 1300, 
                    verbose = 1, early.stop.round = 30,
                    watchlist = watchlist, maximize = FALSE,
                    feval = RMPSE )

pred2 <- exp(predict(model2, data.matrix(test[,-c(2)])))-1


# Averaging
pred <- (pred1+pred2)/2
submission <- data.frame(Id=test$Id, Sales=pred)
cat("saving the submission file\n")
write.csv(submission, "gbm1.csv", row.names = FALSE)

