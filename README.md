# Congressional Bill Modeling
## Samuel Sherman

### Data
Web scraped bill text from congress.gov and retreived bill and votes data using the Sunlight Foundation API.  Bill and vote information dates back to congress 103.

### Non-Negative Matrix Facotrization
Applied NNMF to derive latent topics in bill text and examine any distinguishable characteristics between time periods in congress.

### Regression
With the hypothesis that there is a high level of polarization in government and that these characteristics are prophetic, applied gradient boosted and random forest ensemble methods to predict the percent of yes votes for a given party.

### Classification
With less than 6% of bills introduced in congress since 1993 being brought to a vote, applied random forest, gradient boosted, neural network, and logistic regression models to predict this rare event.

### Python Tools
pymongo, mongodb, sci-kit learn, scipy, numpy, nltk, AWS, BeautifulSoup
