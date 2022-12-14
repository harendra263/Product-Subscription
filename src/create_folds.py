import pandas as pd
import numpy as np
from sklearn import model_selection


if __name__ == "__main__":
    df = pd.read_excel("input/data.xlsx")
    print(df.columns)

    # create a new column called kfold and fill it with -1
    df['kfold'] = -1

    # Randomizing the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    #fetch target
    y = df.y.values

    # initiate the kfold class from model_selection module

    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the kfold column

    for fold, (train, valid) in enumerate(kf.split(X=df, y=y)):
        df.loc[valid, 'kfold']= fold
    
    # save the new csv with kfold column
    df.to_csv("input/train_folds.csv", index=False)