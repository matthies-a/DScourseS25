library(tidyverse)
library(jsonlite)

dates <- url("https://vizgr.org/historical-events/search.php?format=json&begin_date=00000101&%20end_date=20240209&lang=en")

mylist <- fromJSON(dates)
mydf <- bind_rows(mylist$result[-1])
class(mydf$date)
head(mydf)
