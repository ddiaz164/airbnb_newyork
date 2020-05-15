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

But because each host entered their neighborhood for the listing, the official names of the neighborhoods weren’t always used and therefore I could not connect my geojson file to the neighborhood column in order to fill in the price for those boundaries.

## Model Data
When looking at the data, I found there was a lot of information contained in the listing name column but since all the entries were strings I had to transform them into numbers in such a way where it would reflect the information each string contained. To do this I vectorized the name column for a total of about 8000 features, and then used non-negative matrix factorization to find 100 latent topics. I then created 100 new columns (one for each topic), and each column was filled with the probability that the original listing name belonged to that latent topic. So for each listing, the sum of those 100 columns added up to 1.
### Time to train!
Having done all that, I was ready to start training some models. My X matrix contained things like location, review information, and the 100 columns that reflected the name column. And my target, of course, was my price column.
## Initial Models
The biggest mistake I made when looking at the distribution of my target values, was to zoom in on the wrong axis.
![](https://github.com/ddiaz164/airbnb_newyork/blob/master/images/price_dist.PNG)

I saw all my prices clustered around 0-1000 so I zoomed my x-axis to get a better look.
![](https://github.com/ddiaz164/airbnb_newyork/blob/master/images/price_initial.png)

Looking in on the zoom I thought there were just a few above $500 but nothing that would throw me off (wrong).
### Poor Models
I trained my first couple of models and my R squared scores were horrible. 

<code>Binary: </code>  
<code>GradientBoosting   MAE: 65.082 | R<sup>2</sup>: 0.134 </code><br>
<code>RandomForest       MAE: 74.833 | R<sup>2</sup>: 0.092 </code><br>
<code>Probabilities: </code><br>
<code>GradientBoosting   MAE: 58.489 | R<sup>2</sup>: 0.171 </code><br>
<code>RandomForest       MAE: 59.852 | R<sup>2</sup>: 0.153 </code>

I saw some improvement with the probability columns rather than just binary columns but it was still nothing to reliably predict on. This is where I started to panic a little bit thinking I wouldn’t be able to predict price from the information I had alone.
## Target Adjustment
I went back, zoomed in on the y-axis, and saw there was actually a lot more higher priced listings than what I originally saw. Some even as high as $10,000. 

![](https://github.com/ddiaz164/airbnb_newyork/blob/master/images/price_zoom.PNG)
![](https://github.com/ddiaz164/airbnb_newyork/blob/master/images/listing_name.PNG)

I looked at these listings with crazy high prices and saw that the names were not very descriptive. Something like 1 bedroom Lincoln Center costing $10,000 was a hard correlation to make. I did a little digging and found that in some cases, hosts will set a very high price on the listing just to stand out, but will then negotiate that price down once contacted. These listings would be very hard to predict and would damage my model’s ability to predict, so I decided to only look at the data that had prices lower than $500. I would get rid of any anomalies and still retain 94% of my data.


