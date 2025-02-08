import pandas as pd
from sklearn.model_selection import train_test_split
from utils.get_prompt import generate_prompt, generate_test_prompt
from datasets import Dataset
def get_data(filename='dataset/all-data.csv'):
    df = pd.read_csv(filename,
                    names=["sentiment", "text"],
                    encoding="utf-8", encoding_errors="replace",
                    skiprows=1)

    train_data = list()
    test_data = list()
    for sentiment in ["positive", "neutral", "negative"]:
        train, test  = train_test_split(df[df.sentiment==sentiment],
                                        train_size=300,
                                        test_size=300,
                                        random_state=42)
        train_data.append(train)
        test_data.append(test)

    train_data = pd.concat(train_data).sample(frac=1, random_state=10)
    test_data = pd.concat(test_data)

    eval_idx = [idx for idx in df.index if idx not in list(train.index) + list(test.index)]
    eval_data = df[df.index.isin(eval_idx)]
    eval_data = (eval_data
            .groupby('sentiment', group_keys=False)
            .apply(lambda x: x.sample(n=50, random_state=10, replace=True)))
    train_data = train_data.reset_index(drop=True)


    train_data = pd.DataFrame(train_data.apply(generate_prompt, axis=1),
                        columns=["text"])
    eval_data = pd.DataFrame(eval_data.apply(generate_prompt, axis=1),
                        columns=["text"])

    y_gt = test_data.sentiment
    test_data = pd.DataFrame(test_data.apply(generate_test_prompt, axis=1), columns=["text"])

    train_data = Dataset.from_pandas(train_data)
    eval_data = Dataset.from_pandas(eval_data)
    return train_data, eval_data, test_data, y_gt
