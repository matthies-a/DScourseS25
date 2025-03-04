library(rvest)
library(tidyverse)

###QUESTION 3###

url <- "https://www.mlb.com/prospects/stats/top-prospects?type=all&dateRange=Year2024&minPA=1"

webpage <- read_html(url)

data <- webpage %>% html_element("#battingProspects > table") %>% html_table()
data


###QUESTION 4###
library(httr)

endpoint = "series/observations"
params = list(
  api_key= Sys.getenv("FRED_API_KEY"), # API from my environment
  file_type="json", 
  series_id="M2SL"                     #Retrieving M2 money supply
)

fred = 
  httr::GET(
    url = "https://api.stlouisfed.org/",
    path = paste0("fred/", endpoint),   
    query = params                      
  )

fred = 
  fred %>% 
  httr::content("text") %>%
  jsonlite::fromJSON()  

fred =
  fred %>% 
  purrr::pluck("observations") %>%
  as_tibble()
fred
