library(rvest)
library(tidyverse)

#original data scrape

url <- "https://www.mlb.com/prospects/stats/top-prospects?type=all&dateRange=Year2024&minPA=1"

webpage <- read_html(url)

data <- webpage %>% html_element("#battingProspects > table") %>% html_table()
data

df <- as.data.frame(data)

#Removing null values from column
df2 <- subset(df, select = -c(Tm) )
df_new <- df2[1:(length(df2)-3)]

#remving null rows
row_odd <- seq_len(nrow(df_new)) %% 2
MLBdf <- df_new[row_odd == 0, ]

#Rename Columns
colnames(MLBdf)[19:21]=c("HRPercentage","WalkPercentage","StrikeoutPercentage")

library(datasets)
library(ggplot2)

#scatterplotof HR and walk percentages; players further to the top right are top performers
qplot(HRPercentage, WalkPercentage, data = MLBdf)


#add names to the plot to gain more information on players, but data is too dense
ggplot(MLBdf, aes(x = HRPercentage, y = WalkPercentage)) +
  
  geom_point() +
  
  geom_text(aes(label = Player), nudge_x = 1, nudge_y = 1)  

#use ggrepel package to fit data better, still too dense
library(ggrepel)

ggplot(MLBdf, aes(x = HRPercentage, y = WalkPercentage, label = Player)) +
  
  geom_point() + 
  
  geom_text_repel() 

#Create a subset of the data because original data is too dense
subset_data <- MLBdf[1:29,]

ggplot(subset_data, aes(x = HRPercentage, y = WalkPercentage, label = Player)) +
  
  geom_point() + 
  
  geom_text_repel() + labs(title = "Top Performers Amongst MLB Prospects 2024")
