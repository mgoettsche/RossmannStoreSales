## How long has the competition been around?

# Table
table(stores_raw$CompetitionOpenSinceYear, useNA="always")


## Promo Intervals

# Table
table(stores_raw$PromoInterval, useNA="always")

# Are there dirty entries, i.e. shops participating in promos without having an interval set?
nrow(subset(stores_raw, Promo2 != 0 & PromoInterval == ''))

# Or the other way around?
nrow(subset(stores_raw, Promo2 == 0 & PromoInterval != ''))

## Single stores' sales by weekday
store100 <- train_raw[train_raw$Store == 100,]
qplot(Date, Sales, data=store100[store100$DayOfWeek==2,], color=Promo)
