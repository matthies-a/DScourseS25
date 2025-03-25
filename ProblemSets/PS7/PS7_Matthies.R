library(modelsummary)
library(mice)
library(Amelia)
library(tidyr)
library(xtable)

#load in data and convert to data frame
setwd("C:\\Users\\matth\\OneDrive - University of Oklahoma\\DScourseS25\\ProblemSets\\PS7")
wages <- read.csv("wages.csv")
wagedf <- data.frame(wages)

#dropping na in hgc and tenure columns
wagedf2 <- wagedf %>% drop_na(hgc) %>% drop_na(tenure)

##College and married need to be turned into binary variables?

#LaTeX
wage.table<- summary(wagedf2)
print(xtable(wage.table), type = "latex")


##########Listwise deletion###############
wagedf3 <- na.omit(wagedf2)
MCARlm <- lm(logwage ~ hgc + college + tenure + 
               I(age^2) + married, data=wagedf3)

modelsummary(MCARlm)

#############Mean imputation##############
wagedf_meanimpute <- wagedf2 %>% replace_na(list(logwage = mean(wagedf3$logwage)))
Meanimputelm <- lm(logwage~ hgc + college + tenure + I(age^2) + married, 
             data=wagedf_meanimpute)

modelsummary(Meanimputelm)


#########Predicted value MAR#############
wagedf2$logwage[is.na(wagedf2$logwage)] <- predict(
  wagelm, newdata= list(hgc=13.1, college="college grad", tenure=6,age=39,married="married"))
#replaced missing logwages w/ mean inputs for predicted value MAR

Predictedvaluelm <- lm(logwage~ hgc + college + tenure + I(age^2) + married, 
                   data=wagedf2)

modelsummary(Predictedvaluelm)

##########Multiple imputation###########
wage.imp <- mice(wagedf, m = 5, method = 'pmm')
mod[['Listwise deletion']] <- lm(logwage~ hgc + college + tenure + I(age^2) + married, wagedf)
mod[['Mice']] <- with(wage.imp, lm(logwage~ hgc + college + tenure + I(age^2) + married))
mod[['Mice']] <- mice::pool(mod[['Mice']])
modelsummary(mod)


models <- list(
  "MCARlm" = lm(logwage ~ hgc + college + tenure + 
                  I(age^2) + married, data=wagedf3),
  "MeanImputelm" = lm(logwage~ hgc + college + tenure + I(age^2) + married, 
                       data=wagedf_meanimpute),
  "PredictedValuelm" = lm(logwage~ hgc + college + tenure + I(age^2) + married, 
                          data=wagedf2))
modelsummary(models)
modelsummary(models, output = "latex")
###I was having trouble getting the multiple imputation model in the combined
###modelsummary table

###the output of the text code from R was not very neat after putting into LaTeX,
###so I got Claude to help create neater tabular data

