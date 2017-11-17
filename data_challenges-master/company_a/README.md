# A Programming Challenge

## How the project is structured

I broke out the task into a series of subtasks, each in their own directory.  The subtasks are as follows:

* Data Cleaning, found in `data_cleaning`

* Data Resampling, found in `data_resampling`

* Data Understanding, found in `data_understanding`

* Data modeling, found in `data_modeling`

## The Solution

The solution is in a jupyter notebook at the top level directory called `solution.ipynb`.  It is the accumlation of the work, starting with data understanding.  I felt the preprocessing steps weren't germaine to the analysis and so I excluded them from the `solution.ipynb` file.  That work can be found in `data_cleaning`.  

In order to run the notebook please use: 

`jupyter notebook --NotebookApp.iopub_data_rate_limit=10000000000`

Because there are a lot of visualizations and the rate_limit was hit for the default limit.
