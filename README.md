# fakenews-detection
fake news detection by written style

## model-detection.joblib
With this RF model we want to obtain the probability that a political tweet is false news according to its writing, that is to say, taking into account the syntactic content and the emotions it contains.

The detection model was trained on a set of Spanish-language political tweets, categorised as hoaxes or non-hoaxes based on their appearance on hoax detection websites such as Newtral or Maldita.

## characteristics extraction and hoax detection 
In order to detect whether or not a tweet is a hoax, we obtain information and syntactic, semantic, lexical and emotional characteristics of the texts. All these emotions will be used as input arguments in the detection model.
The detection model will return the probability that it does or does not speak about a hoax.

```
python3 characteristics_and_hoax_detection.py -i INPUT-CSV-PATH -o OUTPUT-CSV-PATH -rf RF-PATH -c TEXT-COL
```

Input arguments:
- (`INPUT-CSV-PATH`) csv path to process
- (`OUTPUT-CSV-PATH`) output csv path
- (`RF-PATH`) path RF model (model-detection.joblib)
- (`TEXT-COL`) text column name in input csv

Example:
```
python3 characteristics_and_hoax_detection.py -i /home/tweets_eleccions.csv -o /home/tweets_eleccions_rf.csv -rf /home/model-detection.joblib -c text
```
