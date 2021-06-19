# My model contains 5 features:

1. If a re-tweet is written in the form of a copy-paste (Trump had a tendency to do such retweets).
2. If the tweet contained a link within it (it's employees had a tendency to contain links in their tweets).
3. Each word in the corpus was given a weight that attributed to it its level of prevalence in Trump's tweets compared to his staffer's tweets, so I basically summed up the amount of his email weights for each tweet.
4. In the same form of section 3 every hour of the day gained weight according to the frequency between the two types of tweeters.
5. Trump's employees had a tendency to include in their tweets hours of the type - (number) (am | pm).


At the end of building the features, I trained the model and measured its success using the accuracy index and f1 - the results were pretty good :). Feel free to check me out.
