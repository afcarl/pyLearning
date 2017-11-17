# Priority List

1) 

Subpart A - Do we have a biased Oracle?  Aka is our data biased?

FairML style analysis - look at Age, skill level and any other variables that could potentially indicate demographic bias.

Answer the following question:

Is the decider biased towards or against a certain request based on a demographic feature alone?

How to test for this:

Develop similarity measures for everything except the variable in question and then find records that are similar except in the variable of interest.  Compare the decision chosen for the applicant.  Then run the model excluding the records in question - see if the model predicts same way as the decider.  If the model predicts the same thing as the decider than the model is biased, if the model chooses a different thing than the decider, the decider is biased.

Do this in an IPython Notebook - so you can explain as you go.

Subpart B - Implement GBCENT in Python (use scikit learn if possible).

2) 

Implement MLP NN in:

* tensorflow, 
* PyTorch, 
* Keras.

3) 

* Feature Engineering w/ scikit-feature and semantic feature sets.

* Make use of AutoML in h2o.

4) 

Data visualization w/:
* seaborn, 
* plotly, 
* dash, 
* yellowbrick, 
* d3, 
* c3
* d3plus
* matplotlib
(This maybe overkill, consider a subset of this.)

5) 

* Build a service for your classifier, accepting new data via a API and passing back answers.

* Build a dashboard, showing what metrics you would track as decision points for when to throw out the model.
(Also do this part in IPython Notebook, so you can explain all the metrics you chose.)
