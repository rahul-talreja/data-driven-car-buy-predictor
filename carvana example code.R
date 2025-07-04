#######################
# Logistic Regression #
####################### 

## Load Packages (if needed) and Set Seed

set.seed(1)

## Read in Logistic Regression data

logit <- read.csv(file.choose()) ## Choose training.csv file

## Transform and Create Data

## Dealing with Variables that have NULL Values
logit$MMRCurrentAuctionAveragePrice <- as.numeric(ifelse(logit$MMRCurrentAuctionAveragePrice == "NULL", 0, logit$MMRCurrentAuctionAveragePrice))

## Dealing with Variables that have outliers
logit$lnVehBCost <- log(logit$VehBCost + 1)

## Dealing with Variables that have different categories in training and test
logit$blue = ifelse(logit$Color == "BLUE",1,0)

logit$TrimLX = ifelse(logit$Trim == "LX",1,0)

logit$ModelAVALON3.5LV6EFI = ifelse(logit$Model == "AVALON 3.5L V6 EFI",1,0)

## Run Logistic Regression using GLM

logit_result <- glm(formula = IsBadBuy ~ VehicleAge + MMRCurrentAuctionAveragePrice + lnVehBCost + blue + 
	as.factor(Auction) + as.factor(PRIMEUNIT) + TrimLX + ModelAVALON3.5LV6EFI, data = logit, family = "binomial")
summary(logit_result)

## Pseudo R-square - McFadden

null_result <- glm(formula = IsBadBuy ~ 1, data = logit, family = "binomial")
1 - logLik(logit_result)/logLik(null_result)

## Predicted Probability

logit$predict <- predict(logit_result, logit, type = "response")

## This is just the decile chart of probabilities
quantile(logit$predict, prob = seq(0,1,length=11), type=5) 

## Hit Rate Table

logit$pIsBadBuy <- ifelse(logit$predict >= 0.5, 1, 0)
hitrate <- table(logit$IsBadBuy, logit$pIsBadBuy)
hitrate
sum(diag(hitrate))/sum(hitrate)

## Read in holdout data

logit_test <- read.csv(file.choose()) ## Choose test.csv file

## Transform and Create Data - Create same variables as above; just add _test to table name

logit_test$MMRCurrentAuctionAveragePrice <- as.numeric(ifelse(logit_test$MMRCurrentAuctionAveragePrice == "NULL", 0, logit_test$MMRCurrentAuctionAveragePrice))
logit_test$lnVehBCost <- log(logit_test$VehBCost + 1)
logit_test$blue = ifelse(logit_test$Color == "BLUE",1,0)
logit_test$TrimLX = ifelse(logit_test$Trim == "LX",1,0)
logit_test$ModelAVALON3.5LV6EFI = ifelse(logit_test$Model == "AVALON 3.5L V6 EFI",1,0)

## Predicted Probability for Holdout

logit_test$predict <- predict(logit_result, logit_test, type = "response")
logit_test$IsBadBuy <- ifelse(logit_test$predict >= 0.5, 1, 0)
table(logit_test$IsBadBuy) ## This shows you how many 0s and 1s you predicted

## Create Entry Table for Kaggle

example_entry <- logit_test[c("RefId", "IsBadBuy")]

## Export Logistic Regression Results 

write.csv(example_entry, file = file.choose(new=TRUE), row.names = FALSE) ## Name file example_entry.csv


