# Best Score ~0.1207

### Libraries ###
library(randomForest)
library(caret)
library(lubridate)
library(doMC)

### Globals ###
set.seed(321)
registerDoMC(cores = 6)

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


### Missing Data ###
# CompetitionDistance has 3 NAs, replace with median
stores$CompetitionDistance[is.na(stores$CompetitionDistance)] <- median(stores$CompetitionDistance,
                                                                        na.rm = TRUE)

# CompetitionOpenSinceYear/CompetitionOpenSinceMonth has 354 missing values.
# Replace them with median for the moment, probably change later. Remove outlier
# before calculating median.
stores$CompetitionOpenSinceYear[stores$CompetitionOpenSinceYear==1900] <- NA
stores$CompetitionOpenSinceYear[is.na(stores$CompetitionOpenSinceYear)] <- median(stores$CompetitionOpenSinceYear,
                                                                                  na.rm = TRUE)
stores$CompetitionOpenSinceMonth[is.na(stores$CompetitionOpenSinceMonth)] <- median(stores$CompetitionOpenSinceMonth,
                                                                                  na.rm = TRUE)

# Some 11 test data entries of store 622 have Open==NA. Assume Open==1, because there were
# no holidays.
test$Open[is.na(test$Open)] <- 1

# Promo2SinceWeek and Promo2SinceYear can hardly be imputed. Ignore them in model building
# for the moment.

### Join Train/Test and Stores ###
train <- merge(train, stores)
test <- merge(test, stores)

### Use only data of open days and with sales ###
train <- train[ which(train$Open=='1'),]
train <- train[ which(train$Sales!='0'),]

### Write processed data to CSV files for external models ###
write.table(train, "trainProcessed.csv", row.names=FALSE, quote=FALSE, sep = "::")
write.table(test, "testProcessed.csv", row.names=FALSE, quote=FALSE, sep = "::")


### Modelling ###
print(paste("Training Model:", Sys.time()))
model <- randomForest(train[,-c(6,7, 18, 19)], log(train$Sales+1), mtry=7,
                      ntree=150, sampsize=250000, do.trace = TRUE)
print(paste("Done Training:", Sys.time()))
saveRDS(model, "model.rds")

### Submission ###
prediction <- exp(predict(model, test[,-c(17,18)]))-1
submission <- data.frame(Id=test$Id, Sales=prediction)
write.csv(submission, "submission_rf_150_7.csv", row.names = FALSE)


