#Read datasets
algae <- read.table("Analysis.txt", 
                    header=F, dec='.', col.names=c('season','size','speed','mxPH','mnO2','Cl', 'NO3','NH4','oPO4','PO4','Chla','a1','a2','a3','a4', 'a5','a6','a7'),
                    na.strings=c('XXXXXXX'))
dim(algae)
names(algae)
str(algae)
attributes(algae)
table(algae$season, algae$mxPH)

#Summary
summary(algae)
quantile(algae$mxPH, na.rm=TRUE)
var(algae$mxPH)

#Skewness
install.packages("moments")
library("moments")
hist(algae$mxPH, main ="MXPH")
skewness(algae$mxPH)

#histogram
hist(algae$mxPH, prob = T)

par(mfrow=c(1,2)) # for plotting 2 graphs in one screen
hist(algae$mxPH, prob=T, xlab='',main='Histogram of maximum pH value',ylim=0:1)
lines(density(algae$mxPH,na.rm=T))
rug(jitter(algae$mxPH))

table(algae$season)
pie(table(algae$season))
barplot(table(algae$season))


#Check relationships between 2 variables
cov(algae$a4, algae$a5)
cor(algae[,15:16])

#QQPlots
install.packages("car")
library(car)
qqPlot(algae$mxPH,main='Normal QQ plot of maximum pH')

#boxplots
par(mfrow=c(1,1))
boxplot(algae$oPO4, ylab = "Orthophosphate (oPO4)")
rug(jitter(algae$oPO4), side = 2) 
abline(h = mean(algae$oPO4, na.rm = T), lty = 2)

#scatterplots
plot(algae$mxPH, algae$mnO2, col=algae$season, pch=as.numeric(algae$season))
plot(jitter(algae$mxPH), jitter(algae$mnO2))

pairs(algae[15:16])


install.packages("scatterplot3d")
library(scatterplot3d)
scatterplot3d(algae$mnO2, algae$mxPH, algae$NO3)


#Heatmaps
distMatrix <- as.matrix(dist(algae[,13:16]))
heatmap(distMatrix)

#Data Matrix alternative
algae_matrix <- as.matrix(algae[,13:16])
image(algae_matrix)


#parallel coordinates
install.packages("MASS")
library(MASS)
parcoord(algae[13:16], col=algae$season)

#Proximities: Data Similiarity & Dissimilarity
dist(algae[13:16,], method="euclidean")
dist(algae[13:16,], method="manhattan")
dist(algae[13:16,], method="maximum")

#Distances for binary data
# 1. Jaccard Distance
dist(algae[13:16,], method="binary")
# 2.  Hamming Distance
dist(algae[13:16,], method="euclidean")^2

#outliers
plot(algae$NH4, xlab = "")
abline(h = mean(algae$NH4, na.rm = T), lty = 1) #plot with mean
abline(h = mean(algae$NH4, na.rm = T) + sd(algae$NH4, na.rm = T),lty = 2)#mean + Standard deviation
abline(h = median(algae$NH4, na.rm = T), lty = 3)
identify(algae$NH4)

plot(algae$NH4, xlab = "")
clicked.lines <- identify(algae$NH4)
algae[clicked.lines, ]