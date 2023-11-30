# k-nearest-neighbors

My python implementation of the k-nearest neighbors algorithm for the Iris dataset. Originally done on a Kaggle
Jupyter notebook. 

The data set used for training is the classic one of Fisher from 1936 (Fisher, 1988) with 150 samples and can be found on the 
UCI Machine Learning Repository. The implementation was tested by using an extended version of the iris data set by Baladram (2023) with 1200 samples. By adjusting manually the value of k, it was found that the highest number of correct guesses, 81.9%, was obtained when k = 1. Similar results where observed when k = 4, with a success rate of 81.5%. 

### Known issues and possible improvements
* There are several grammar and spelling errors in some variable names.
* Code is still not PEP8 compliant.
* The method classify in the Model class does not break ties between the number of neighbors.


## Sources
* Fisher,R. A.. (1988). Iris. UCI Machine Learning Repository. https://doi.org/10.24432/C56C76
* Samy Baladram. (2023). <i>🌺 Iris Dataset Extended</i> [Data set]. Kaggle. https://doi.org/10.34740/KAGGLE/DS/3916784
