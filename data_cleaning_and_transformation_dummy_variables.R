# ARCHIVED, replacing categorical features with binary dummies makes no sense here.

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

# Expand StateHoliday to binary features
StateHolidayExpanded <- model.matrix(~ StateHoliday - 1, data=train)
train <- cbind(train, StateHolidayExpanded)
StateHolidayExpanded <- model.matrix(~ StateHoliday - 1, data=test)
test <- cbind(test, StateHolidayExpanded)

# Remove un-expanded columns
columnsToRemove <- c("Date", "StateHoliday")
train <- train[,!(names(train) %in% columnsToRemove)]
test <- test[,!(names(test) %in% columnsToRemove)]

# Re-order columns for visual inspection
# StateHoliday == (b|c) not present in test set, thus removing the columns.
# In real-word application you would want to keep them.
ordering_train <- c("Store", "Year", "Month", "Day", "DayOfWeek", "Sales", "Customers", "Open", "Promo",
              "SchoolHoliday", "StateHoliday0", "StateHolidaya", "StateHolidayb", "StateHolidayc")
ordering_test <- c("Id", "Store", "Year", "Month", "Day", "DayOfWeek", "Open", "Promo",
              "SchoolHoliday", "StateHoliday0", "StateHolidaya")
train <- train[,ordering_train]
test <- test[,ordering_test]


### Feature Engineering: Stores ###
# Expand StoreType to binary features
StoreTypeExpanded <- model.matrix(~ StoreType + Assortment - 1, data=stores_raw)
stores <- cbind(stores_raw, StoreTypeExpanded)

# Expand Assortment to binary features
AssortmentExpanded <- model.matrix(~ Assortment - 1, data=stores)
stores <- cbind(stores, AssortmentExpanded)

# Expand PromoInterval to binary features
months <- c("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sept", "Oct", "Nov", "Dec")

PromoIntervalExpanded <- model.matrix(~ PromoInterval - 1, data=stores)
stores <- cbind(stores, PromoIntervalExpanded)

# Remove un-expanded columns
columnsToRemove <- c("StoreType", "Assortment", "PromoInterval")
stores <- stores[,!(names(stores) %in% columnsToRemove)]

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
train <- train[ which(train$Open=='1'),]
test <- merge(test, stores)
