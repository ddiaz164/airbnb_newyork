![](https://tophotel.news/wp-content/uploads/2019/03/New-York-City-Brooklyn-Bridge-Panorama-Juergen-Roth-2.jpg)
# New York City Airbnb Listings: Predicting the Price
New York City is the nation’s largest short-term rental market and it is the largest domestic market for Airbnb with over 50,000 apartment rental listings. However, Airbnb does not permit hosts to have more than one listing at any single address in New York City so each listing pertains to one unique address. 
## Data Source
I found the data I used on Kaggle, it describes about ten different listing activities and metrics for New York City in 2019 and it has almost 50,000 listings to look at. Since each listing belongs to a unique address, the location metrics contained in the data set would prove very beneficial in a model.
## Data Preparation
As a pleasant surprise, the dataset came mostly clean with only a few columns having NaNs and they were very easy to fill according to their values. The reviews and reviews per month seemed to have NaNs when the listing had not been reviewed yet so those I filled with 0, and the name/host name columns have some NaNs so those I filled with 'Unknown'.
## Initial Lookthrough
The first thing I wanted to do was to take a look at how these listings were distributed across the city so I decided to make a heat map of airbnb counts throughout the city. 
![](https://github.com/ddiaz164/airbnb_newyork/blob/master/images/heat_map.png)

I found that most of these listings were in the Manhattan and Brooklyn area with a small number of them being in Bronx and Staten Island, and a moderate amount in Queens.
## Price Distribution
To get a better visual of the price ranges in the city, I grouped each borough on average price and was not surprised to find that the more dense boroughs had a higher average price per listing. 
![](https://github.com/ddiaz164/airbnb_newyork/blob/master/images/choro_boroughs.PNG)
In a similar way, I wanted to look at the average price per neighborhood. 
![](https://github.com/ddiaz164/airbnb_newyork/blob/master/images/choro_wrong.PNG)
But because each host entered their neighborhood, the official names of the neighborhoods weren’t always used and therefore I could not connect my geojson file to the neighborhood column.
