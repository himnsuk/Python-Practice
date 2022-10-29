from sklearn import preprocessing 


"""
- label encoding
- one hot encoding
- binarization
"""

class CategoricalFeatures:
  def __init__(self, df, categorical_features, encoding_type, handle_na=False):
    """
    df: pandas dataframe
    categorical_features: list of column names, e.g. ["ord_1", "nom_0"....]
    encoding_type: label, binary, onehotencoding
    """
    self.df = df
    self.output_df = self.df.copy(deep = True)
    self.cat_feats = categorical_features
    self.enc_type = encoding_type
    self.label_encoder = dict()

    if handle_na:
      for c in self.cat_feats:
        self.df.loc[:, c] = self.df.loc[:, c].astype(str).fillna("-99999")
     
  
  def _label_encoding(self):
    for c in self.cat_feats:
      lbl = preprocessing.LabelEncoder()
      lbl.fit(self.df[c].values)
      self.output_df.loc[:, c] = lbl.transform(self.df[c].values)
      self.label_encoder[c] = lbl
    return self.output_df

  def transform(self):
    if self.enc_type == 'label':
     return self._label_encoding() 
    else:
      raise Exception("Encoding type not understood")
  



if __name__ == "__main__":
  import pandas as pd
  df = pd.read_csv("https://raw.githubusercontent.com/r12king/Kaggle-Categorical-Feature-Encoding-Challenge-II/develop/data/raw/train.csv")
  # cols = [c in df.columns if c not in ["id", "target"]]
  cols = [c for c in df.columns if c not in ["id", "target"]]
  print(cols)
  cat_feats = CategoricalFeatures(df, 
                                  categorical_features= cols,
                                  encoding_type='label',
                                  handle_na = True)
  output_df = cat_feats.transform()
  
  print(output_df.head())