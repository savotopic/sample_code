library(shiny)
library(readr)
library(corrplot)
library(cowplot)
library(irtoys)
library(plsdepot)
library(ggplot2)
library(visreg)
library(caret)

re_boston = read.csv("zillow-boston.csv", header = TRUE, sep = (","))

re_boston$property_id = NULL
re_boston$address = NULL
re_boston$street_name = NULL
re_boston$latitude = NULL
re_boston$longitude = NULL
re_boston$property_status = NULL
re_boston$agency_name = NULL
re_boston$zillow_owned = NULL
re_boston$RunDate = NULL

#I'm interested only in condo data
re_boston_ma = subset(re_boston, rstate == "MA" & property_type == "CONDO" & bathroom_number < 4 & bathroom_number > 0 & bedroom_number < 4 & bedroom_number > 0 & price < 2.5e+6)

#Clean data additionally - the datase now only includes condo data from Boston MA
re_boston_ma$city = NULL
re_boston_ma$rstate = NULL
re_boston_ma$property_type = NULL
re_boston_ma$land_space = NULL

re_boston_ma$price = re_boston_ma$price/1e+6

#nrow(re_boston_ma)
#nrow(re_boston_ma)

str(re_boston_ma)
summary(re_boston_ma)

plot.new()
par(mfrow = c(1, 3))
boxplot(re_boston_ma$price, main='Price',col='Red')
boxplot(re_boston_ma$zestimate, main='Zestimate',col='Red')
boxplot(re_boston_ma$price_per_unit, main='Price per Unit',col='Red')

#Investigate correlations
# plot.new()
# corrplot(cor(re_boston_ma, use="complete.obs"));
# corrplot = recordPlot();

corr1 = cor(re_boston_ma$price, re_boston_ma$bathroom_number, use = "complete.obs");
plot(re_boston_ma$price, re_boston_ma$bathroom_number, pch = 19, col = "black", main = 'Price vs Number of Bathrooms', cex.main=2);
text(paste("Correlation:", round(corr1, 2)), x = 0.5, y = 6, cex= 2.5);
price_bath_plot = recordPlot();
plot.new();
frame()


#plot.new()
corr2 = cor(re_boston_ma$price, re_boston_ma$living_space, use = "complete.obs");
plot(re_boston_ma$price, re_boston_ma$living_space, pch = 19, col = "black", main = 'Condo Price vs Living Space', cex.main=2);
text(paste("Correlation:", round(corr2, 2)), x = 0.5, y = 3000, cex= 2.5);
price_space_plot = recordPlot();
plot.new()
frame()


# corr = cor(re_boston_ma$price, re_boston_ma$bathroom_number, use = "complete.obs")
# text(paste("Correlation:", round(corr, 2)), x = 1.5, y = 4.5)

about_text_file = read_file("about.txt")

### Split the data as Training and Test sets
set.seed(2024)
splitBos = caret::createDataPartition(re_boston_ma[,1], p = 0.8, list=F, times=1)
splitBos
trainBos = re_boston_ma[splitBos,]
head(trainBos)
testBos = re_boston_ma[!row.names(re_boston_ma) %in% row.names(trainBos),]
testBos = re_boston_ma[-splitBos,]
testBos

### Aplying Linear Regression model
lr = lm(price ~ bedroom_number + bathroom_number + living_space, data=trainBos[,c(2,4,5,7)])
fitted(lr)
resid(lr)
#lr
#summary(lr)

### Predict
predict(lr, testBos[,c(4,5,7)], level=.95, interval="confidence")
predBos = data.frame(predict(lr, testBos[,c(4,5,7)], level=.95, interval="confidence"))
predBos$Reference = testBos[,c(2)]
#qplot(Reference, fit, data=predBos) + geom_point(colour = "#3366FF", size = 3) + geom_errorbar(aes(ymin = lwr,ymax = upr))

#plot.new()
cor(predBos$fit, predBos$Reference)
corr3 = cor(predBos$fit, predBos$Reference, use = "complete.obs");
plot(predBos$fit, predBos$Reference, pch = 19, col = "black", main = 'Predicted Price vs Reference Price', cex.main=2);
text(paste("Correlation:", round(corr3, 2)), x = 0.6, y = 2.2, cex= 2.5);
pred_ref_plot = recordPlot();
plot.new()
frame()

newdf = data.frame(bedroom_number = 4, bathroom_number = 3, living_space = 1153)
price_pred = predict(lr, newdf, level=.95, interval="confidence", response = TRUE)
