# Python Machine Learning OpenCV Teaching Examples

OpenCV Python machine learning examples used for teaching within the undergraduate Computer Science programme
at [Durham University](http://www.durham.ac.uk) (UK) by [Prof. Toby Breckon](http://community.dur.ac.uk/toby.breckon/).

All tested with [OpenCV](http://www.opencv.org) 3.x and Python 3.x.

---

### Background:

Directly adapted from the older [C++](https://github.com/tobybreckon/cpp-examples-ml) OpenCV machine learning teaching examples _(that for a long time, the in absence of other fully worked examples for the OpenCV machine learning components became the defacto reference for the use of these OpenCV routines)_

All dataset examples are taken and reproduced from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/).

A related set of [Python Image Processing OpenCV Teaching Examples](https://github.com/tobybreckon/python-examples-ip.git) are also available covering basic image processing operations and similarly a set of [Python Computer Vision OpenCV Teaching Examples](https://github.com/tobybreckon/python-examples-cv.git)

---

### How to download and run:

Download each file as needed or to download the entire repository and run each try:

```
git clone https://github.com/tobybreckon/python-examples-ml.git
cd python-examples-ml
cd <sub directory of one of the examples>
python3 ./<insert file name of one of the examples>.py
```

-- which _should_ then produce an output of the results on the dataset inside that example directory.

In each sub-directory:

+ .py file(s) - code for the examples (several examples per directory in many cases)
+ .name file - an explanation of the data and its source
+ .data file - the original and complete set of data (CSV file format)
+ .train file - the data to be used for training (CSV file format)
+ .test file - the data to be used for testing (CSV file format)
+ .xml, .yml - if present, example data files for testing some tools

For some examples you may need to copy/link the .train/.test files from one of the other directories (it seemed silly to archive them multiple times).

Demo source code is provided _"as is"_ to aid learning and understanding of topics on the course and beyond.

---

If referencing these examples in your own work please use:
```
@TechReport{breckon2014,
  author =       {Breckon, T.P.},
  title =        {Machine Learning - Course Notes and Materials},
  institution =  {Durham University},
  year =         {2014},
  address =      {Durham, UK},
}
```
---

If you find any bugs raise an issue (or much better still submit a git pull request with a fix) - toby.breckon@durham.ac.uk

_"may the source be with you"_ - anon.
