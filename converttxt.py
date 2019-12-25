import pandas as pd


def convert(txt):
    mycolnames = ['Datetime','Face','Eye','Path']
    df = pd.read_csv(txt, sep="|")
    df.columns = mycolnames
    df.to_csv("test.csv", sep=",")