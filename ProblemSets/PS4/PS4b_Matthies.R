#!/usr/bin/env Rscript

module load Java/11
install.packages("sparklyr")
install.packages("tidyverse")


library(tidyverse)
library(sparklyr)

sc <- spark_connect(master = "local")

df1 <- as.tibble(iris)

df <- copy_to(sc, df1)

class(df1)
class(df)

df %>% select(Sepal_Length,Species) %>% head%>% print %>% filter(Sepal_Length>5.5) %>%head %>% print.

df2 <- df %>% group_by(Species) %>% summarize(mean = mean(Sepal_Length),count = n()) %>% head %>% print

