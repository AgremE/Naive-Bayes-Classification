# Naive Bayes Classification

This is a home from a class CS570 (Artificial Intelligent and Machine Learning).

## Require Package

This particular project needs the following package to run:

	1- Python 2.7
	2- Numpy

## Usage

To try out the classification pipeline, run dataClassifier.py from the command line. This will clas-
sify the digit data using the default classifier (mostFrequent) which blindly classifies every example
with the most frequent label.

	python dataClassifier.py

To activate the naive bayes classifier, use -c naiveBayes:

	python dataClassifier.py -c naiveBayes

Without the option --autotune, kgrid is given to 1 as default. In this case, you can change the
smoothing parameter k with -k. --autotune option makes the classifier to be trained with different
values of k.

	python dataClassifier.py -c naiveBayes --autotune

To run on the face recognition dataset, use -d faces.

	python dataClassifier.py -d faces -c naiveBayes --autotune

	

