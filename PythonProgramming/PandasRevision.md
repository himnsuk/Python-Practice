Pandas Revision Basic Functionalities to remember
----


```python
import pandas as pd
```


```python
pd.__version__
```




    '1.1.5'




```python
df = pd.read_csv("https://raw.githubusercontent.com/himnsuk/2021-07-13-scipy-pandas/main/data/gapminder.tsv", sep="\t")
```


```python
df
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>continent</th>
      <th>year</th>
      <th>lifeExp</th>
      <th>pop</th>
      <th>gdpPercap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1952</td>
      <td>28.801</td>
      <td>8425333</td>
      <td>779.445314</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1957</td>
      <td>30.332</td>
      <td>9240934</td>
      <td>820.853030</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1962</td>
      <td>31.997</td>
      <td>10267083</td>
      <td>853.100710</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1967</td>
      <td>34.020</td>
      <td>11537966</td>
      <td>836.197138</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1972</td>
      <td>36.088</td>
      <td>13079460</td>
      <td>739.981106</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1699</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>1987</td>
      <td>62.351</td>
      <td>9216418</td>
      <td>706.157306</td>
    </tr>
    <tr>
      <th>1700</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>1992</td>
      <td>60.377</td>
      <td>10704340</td>
      <td>693.420786</td>
    </tr>
    <tr>
      <th>1701</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>1997</td>
      <td>46.809</td>
      <td>11404948</td>
      <td>792.449960</td>
    </tr>
    <tr>
      <th>1702</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>2002</td>
      <td>39.989</td>
      <td>11926563</td>
      <td>672.038623</td>
    </tr>
    <tr>
      <th>1703</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>2007</td>
      <td>43.487</td>
      <td>12311143</td>
      <td>469.709298</td>
    </tr>
  </tbody>
</table>
<p>1704 rows × 6 columns</p>
</div>




```python
type(df)
```




    pandas.core.frame.DataFrame



Shape gives you number of rows and column

Shape is an attribute no curly bracket needs


```python
df.shape
```




    (1704, 6)




1. info tells gives us a summary of the columns in our dfaset
2. info is a method (note the round brackets)
3. and you call methods with the dot
4. contrast this with the type function example


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1704 entries, 0 to 1703
    Data columns (total 6 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   country    1704 non-null   object 
     1   continent  1704 non-null   object 
     2   year       1704 non-null   int64  
     3   lifeExp    1704 non-null   float64
     4   pop        1704 non-null   int64  
     5   gdpPercap  1704 non-null   float64
    dtypes: float64(2), int64(2), object(2)
    memory usage: 80.0+ KB



```python
# in ipython / jupyter notebook / jupyter lab
# you can run pwd to "print working directory"
# use this to help you find files you want to load
!pwd
```

    /content



```python
df.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>continent</th>
      <th>year</th>
      <th>lifeExp</th>
      <th>pop</th>
      <th>gdpPercap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1952</td>
      <td>28.801</td>
      <td>8425333</td>
      <td>779.445314</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1957</td>
      <td>30.332</td>
      <td>9240934</td>
      <td>820.853030</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1962</td>
      <td>31.997</td>
      <td>10267083</td>
      <td>853.100710</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1967</td>
      <td>34.020</td>
      <td>11537966</td>
      <td>836.197138</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1972</td>
      <td>36.088</td>
      <td>13079460</td>
      <td>739.981106</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>continent</th>
      <th>year</th>
      <th>lifeExp</th>
      <th>pop</th>
      <th>gdpPercap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1699</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>1987</td>
      <td>62.351</td>
      <td>9216418</td>
      <td>706.157306</td>
    </tr>
    <tr>
      <th>1700</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>1992</td>
      <td>60.377</td>
      <td>10704340</td>
      <td>693.420786</td>
    </tr>
    <tr>
      <th>1701</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>1997</td>
      <td>46.809</td>
      <td>11404948</td>
      <td>792.449960</td>
    </tr>
    <tr>
      <th>1702</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>2002</td>
      <td>39.989</td>
      <td>11926563</td>
      <td>672.038623</td>
    </tr>
    <tr>
      <th>1703</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>2007</td>
      <td>43.487</td>
      <td>12311143</td>
      <td>469.709298</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.sample(30)
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>continent</th>
      <th>year</th>
      <th>lifeExp</th>
      <th>pop</th>
      <th>gdpPercap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1023</th>
      <td>Morocco</td>
      <td>Africa</td>
      <td>1967</td>
      <td>50.335</td>
      <td>14770296</td>
      <td>1711.044770</td>
    </tr>
    <tr>
      <th>1359</th>
      <td>Singapore</td>
      <td>Asia</td>
      <td>1967</td>
      <td>67.946</td>
      <td>1977600</td>
      <td>4977.418540</td>
    </tr>
    <tr>
      <th>1633</th>
      <td>Venezuela</td>
      <td>Americas</td>
      <td>1957</td>
      <td>57.907</td>
      <td>6702668</td>
      <td>9802.466526</td>
    </tr>
    <tr>
      <th>394</th>
      <td>Cuba</td>
      <td>Americas</td>
      <td>2002</td>
      <td>77.158</td>
      <td>11226999</td>
      <td>6340.646683</td>
    </tr>
    <tr>
      <th>1547</th>
      <td>Togo</td>
      <td>Africa</td>
      <td>2007</td>
      <td>58.420</td>
      <td>5701579</td>
      <td>882.969944</td>
    </tr>
    <tr>
      <th>1673</th>
      <td>Yemen, Rep.</td>
      <td>Asia</td>
      <td>1977</td>
      <td>44.175</td>
      <td>8403990</td>
      <td>1829.765177</td>
    </tr>
    <tr>
      <th>691</th>
      <td>Iceland</td>
      <td>Europe</td>
      <td>1987</td>
      <td>77.230</td>
      <td>244676</td>
      <td>26923.206280</td>
    </tr>
    <tr>
      <th>645</th>
      <td>Haiti</td>
      <td>Americas</td>
      <td>1997</td>
      <td>56.671</td>
      <td>6913545</td>
      <td>1341.726931</td>
    </tr>
    <tr>
      <th>627</th>
      <td>Guinea-Bissau</td>
      <td>Africa</td>
      <td>1967</td>
      <td>35.492</td>
      <td>601287</td>
      <td>715.580640</td>
    </tr>
    <tr>
      <th>1211</th>
      <td>Peru</td>
      <td>Americas</td>
      <td>2007</td>
      <td>71.421</td>
      <td>28674757</td>
      <td>7408.905561</td>
    </tr>
    <tr>
      <th>485</th>
      <td>Equatorial Guinea</td>
      <td>Africa</td>
      <td>1977</td>
      <td>42.024</td>
      <td>192675</td>
      <td>958.566812</td>
    </tr>
    <tr>
      <th>66</th>
      <td>Australia</td>
      <td>Oceania</td>
      <td>1982</td>
      <td>74.740</td>
      <td>15184200</td>
      <td>19477.009280</td>
    </tr>
    <tr>
      <th>1573</th>
      <td>Turkey</td>
      <td>Europe</td>
      <td>1957</td>
      <td>48.079</td>
      <td>25670939</td>
      <td>2218.754257</td>
    </tr>
    <tr>
      <th>1358</th>
      <td>Singapore</td>
      <td>Asia</td>
      <td>1962</td>
      <td>65.798</td>
      <td>1750200</td>
      <td>3674.735572</td>
    </tr>
    <tr>
      <th>953</th>
      <td>Mali</td>
      <td>Africa</td>
      <td>1977</td>
      <td>41.714</td>
      <td>6491649</td>
      <td>686.395269</td>
    </tr>
    <tr>
      <th>623</th>
      <td>Guinea</td>
      <td>Africa</td>
      <td>2007</td>
      <td>56.007</td>
      <td>9947814</td>
      <td>942.654211</td>
    </tr>
    <tr>
      <th>1482</th>
      <td>Switzerland</td>
      <td>Europe</td>
      <td>1982</td>
      <td>76.210</td>
      <td>6468126</td>
      <td>28397.715120</td>
    </tr>
    <tr>
      <th>200</th>
      <td>Burkina Faso</td>
      <td>Africa</td>
      <td>1992</td>
      <td>50.260</td>
      <td>8878303</td>
      <td>931.752773</td>
    </tr>
    <tr>
      <th>1059</th>
      <td>Namibia</td>
      <td>Africa</td>
      <td>1967</td>
      <td>51.159</td>
      <td>706640</td>
      <td>3793.694753</td>
    </tr>
    <tr>
      <th>685</th>
      <td>Iceland</td>
      <td>Europe</td>
      <td>1957</td>
      <td>73.470</td>
      <td>165110</td>
      <td>9244.001412</td>
    </tr>
    <tr>
      <th>237</th>
      <td>Cameroon</td>
      <td>Africa</td>
      <td>1997</td>
      <td>52.199</td>
      <td>14195809</td>
      <td>1694.337469</td>
    </tr>
    <tr>
      <th>1389</th>
      <td>Slovenia</td>
      <td>Europe</td>
      <td>1997</td>
      <td>75.130</td>
      <td>2011612</td>
      <td>17161.107350</td>
    </tr>
    <tr>
      <th>1481</th>
      <td>Switzerland</td>
      <td>Europe</td>
      <td>1977</td>
      <td>75.390</td>
      <td>6316424</td>
      <td>26982.290520</td>
    </tr>
    <tr>
      <th>832</th>
      <td>Korea, Dem. Rep.</td>
      <td>Asia</td>
      <td>1972</td>
      <td>63.983</td>
      <td>14781241</td>
      <td>3701.621503</td>
    </tr>
    <tr>
      <th>416</th>
      <td>Denmark</td>
      <td>Europe</td>
      <td>1992</td>
      <td>75.330</td>
      <td>5171393</td>
      <td>26406.739850</td>
    </tr>
    <tr>
      <th>651</th>
      <td>Honduras</td>
      <td>Americas</td>
      <td>1967</td>
      <td>50.924</td>
      <td>2500689</td>
      <td>2538.269358</td>
    </tr>
    <tr>
      <th>714</th>
      <td>Indonesia</td>
      <td>Asia</td>
      <td>1982</td>
      <td>56.159</td>
      <td>153343000</td>
      <td>1516.872988</td>
    </tr>
    <tr>
      <th>657</th>
      <td>Honduras</td>
      <td>Americas</td>
      <td>1997</td>
      <td>67.659</td>
      <td>5867957</td>
      <td>3160.454906</td>
    </tr>
    <tr>
      <th>1396</th>
      <td>Somalia</td>
      <td>Africa</td>
      <td>1972</td>
      <td>40.973</td>
      <td>3840161</td>
      <td>1254.576127</td>
    </tr>
    <tr>
      <th>638</th>
      <td>Haiti</td>
      <td>Americas</td>
      <td>1962</td>
      <td>43.590</td>
      <td>3880130</td>
      <td>1796.589032</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Getting columns of the dfa frame
df.columns
```




    Index(['country', 'continent', 'year', 'lifeExp', 'pop', 'gdpPercap'], dtype='object')




```python
# Getting index of dfa frame
df.index
```




    RangeIndex(start=0, stop=1704, step=1)




```python
# Getting values of dfaframe as a numpy array
df.values
```




    array([['Afghanistan', 'Asia', 1952, 28.801, 8425333, 779.4453145],
           ['Afghanistan', 'Asia', 1957, 30.331999999999997, 9240934,
            820.8530296],
           ['Afghanistan', 'Asia', 1962, 31.997, 10267083, 853.1007099999999],
           ...,
           ['Zimbabwe', 'Africa', 1997, 46.809, 11404948, 792.4499602999999],
           ['Zimbabwe', 'Africa', 2002, 39.989000000000004, 11926563,
            672.0386227000001],
           ['Zimbabwe', 'Africa', 2007, 43.486999999999995, 12311143,
            469.70929810000007]], dtype=object)




```python
# Use square bracket to subset a column

df["country"]
```




    0       Afghanistan
    1       Afghanistan
    2       Afghanistan
    3       Afghanistan
    4       Afghanistan
               ...     
    1699       Zimbabwe
    1700       Zimbabwe
    1701       Zimbabwe
    1702       Zimbabwe
    1703       Zimbabwe
    Name: country, Length: 1704, dtype: object




```python
# A single column will return a series
type(df['country'])
```




    pandas.core.series.Series




```python
# pass in a [list] to [subset] multiple columns
df[["country", "pop"]]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>8425333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>9240934</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>10267083</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>11537966</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>13079460</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1699</th>
      <td>Zimbabwe</td>
      <td>9216418</td>
    </tr>
    <tr>
      <th>1700</th>
      <td>Zimbabwe</td>
      <td>10704340</td>
    </tr>
    <tr>
      <th>1701</th>
      <td>Zimbabwe</td>
      <td>11404948</td>
    </tr>
    <tr>
      <th>1702</th>
      <td>Zimbabwe</td>
      <td>11926563</td>
    </tr>
    <tr>
      <th>1703</th>
      <td>Zimbabwe</td>
      <td>12311143</td>
    </tr>
  </tbody>
</table>
<p>1704 rows × 2 columns</p>
</div>




```python

# if you pass in a list of a single column
# you get a dfaframe back
df[["country"]]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>1699</th>
      <td>Zimbabwe</td>
    </tr>
    <tr>
      <th>1700</th>
      <td>Zimbabwe</td>
    </tr>
    <tr>
      <th>1701</th>
      <td>Zimbabwe</td>
    </tr>
    <tr>
      <th>1702</th>
      <td>Zimbabwe</td>
    </tr>
    <tr>
      <th>1703</th>
      <td>Zimbabwe</td>
    </tr>
  </tbody>
</table>
<p>1704 rows × 1 columns</p>
</div>




```python
# use loc to return rows and match by the actual value in the index
df.loc[0]
```




    country      Afghanistan
    continent           Asia
    year                1952
    lifeExp           28.801
    pop              8425333
    gdpPercap        779.445
    Name: 0, dtype: object




```python
df.loc[338]
```




    country      Congo, Rep.
    continent         Africa
    year                1962
    lifeExp           48.435
    pop              1047924
    gdpPercap        2464.78
    Name: 338, dtype: object




```python
# return multiple rows by passing in a list
df.loc[[0, 779]]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>continent</th>
      <th>year</th>
      <th>lifeExp</th>
      <th>pop</th>
      <th>gdpPercap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1952</td>
      <td>28.801</td>
      <td>8425333</td>
      <td>779.445314</td>
    </tr>
    <tr>
      <th>779</th>
      <td>Italy</td>
      <td>Europe</td>
      <td>2007</td>
      <td>80.546</td>
      <td>58147733</td>
      <td>28569.719700</td>
    </tr>
  </tbody>
</table>
</div>




```python
# -1 does not exist as an index value
# df.loc[-1]
```


```python
# if you want to use index positions, use iloc
df.iloc[-1]
```




    country      Zimbabwe
    continent      Africa
    year             2007
    lifeExp        43.487
    pop          12311143
    gdpPercap     469.709
    Name: 1703, dtype: object




```python
# you can use loc and iloc to subset rows and columns
# [ROW, COLUMN]
df.loc[:, :]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>continent</th>
      <th>year</th>
      <th>lifeExp</th>
      <th>pop</th>
      <th>gdpPercap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1952</td>
      <td>28.801</td>
      <td>8425333</td>
      <td>779.445314</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1957</td>
      <td>30.332</td>
      <td>9240934</td>
      <td>820.853030</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1962</td>
      <td>31.997</td>
      <td>10267083</td>
      <td>853.100710</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1967</td>
      <td>34.020</td>
      <td>11537966</td>
      <td>836.197138</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>Asia</td>
      <td>1972</td>
      <td>36.088</td>
      <td>13079460</td>
      <td>739.981106</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1699</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>1987</td>
      <td>62.351</td>
      <td>9216418</td>
      <td>706.157306</td>
    </tr>
    <tr>
      <th>1700</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>1992</td>
      <td>60.377</td>
      <td>10704340</td>
      <td>693.420786</td>
    </tr>
    <tr>
      <th>1701</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>1997</td>
      <td>46.809</td>
      <td>11404948</td>
      <td>792.449960</td>
    </tr>
    <tr>
      <th>1702</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>2002</td>
      <td>39.989</td>
      <td>11926563</td>
      <td>672.038623</td>
    </tr>
    <tr>
      <th>1703</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>2007</td>
      <td>43.487</td>
      <td>12311143</td>
      <td>469.709298</td>
    </tr>
  </tbody>
</table>
<p>1704 rows × 6 columns</p>
</div>




```python
# select all the rows and the year and population columns
df.loc[:, ["year", "pop"]]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1952</td>
      <td>8425333</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1957</td>
      <td>9240934</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1962</td>
      <td>10267083</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1967</td>
      <td>11537966</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1972</td>
      <td>13079460</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1699</th>
      <td>1987</td>
      <td>9216418</td>
    </tr>
    <tr>
      <th>1700</th>
      <td>1992</td>
      <td>10704340</td>
    </tr>
    <tr>
      <th>1701</th>
      <td>1997</td>
      <td>11404948</td>
    </tr>
    <tr>
      <th>1702</th>
      <td>2002</td>
      <td>11926563</td>
    </tr>
    <tr>
      <th>1703</th>
      <td>2007</td>
      <td>12311143</td>
    </tr>
  </tbody>
</table>
<p>1704 rows × 2 columns</p>
</div>




```python
# select multiple rows and columns based on name using loc
# note ["year", "pop"] will not work if you use iloc,
# because iloc needs index positions
df.loc[[0, 10, 100], ["year", "pop"]]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1952</td>
      <td>8425333</td>
    </tr>
    <tr>
      <th>10</th>
      <td>2002</td>
      <td>25268405</td>
    </tr>
    <tr>
      <th>100</th>
      <td>1972</td>
      <td>70759295</td>
    </tr>
  </tbody>
</table>
</div>




```python
# get a boolean vector/series of a conditional match
df["country"] == "Zimbabwe"
```




    0       False
    1       False
    2       False
    3       False
    4       False
            ...  
    1699     True
    1700     True
    1701     True
    1702     True
    1703     True
    Name: country, Length: 1704, dtype: bool




```python
# use conditional match to subset dfaframe

df.loc[df['country'] == "Zimbabwe", :]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>continent</th>
      <th>year</th>
      <th>lifeExp</th>
      <th>pop</th>
      <th>gdpPercap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1692</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>1952</td>
      <td>48.451</td>
      <td>3080907</td>
      <td>406.884115</td>
    </tr>
    <tr>
      <th>1693</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>1957</td>
      <td>50.469</td>
      <td>3646340</td>
      <td>518.764268</td>
    </tr>
    <tr>
      <th>1694</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>1962</td>
      <td>52.358</td>
      <td>4277736</td>
      <td>527.272182</td>
    </tr>
    <tr>
      <th>1695</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>1967</td>
      <td>53.995</td>
      <td>4995432</td>
      <td>569.795071</td>
    </tr>
    <tr>
      <th>1696</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>1972</td>
      <td>55.635</td>
      <td>5861135</td>
      <td>799.362176</td>
    </tr>
    <tr>
      <th>1697</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>1977</td>
      <td>57.674</td>
      <td>6642107</td>
      <td>685.587682</td>
    </tr>
    <tr>
      <th>1698</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>1982</td>
      <td>60.363</td>
      <td>7636524</td>
      <td>788.855041</td>
    </tr>
    <tr>
      <th>1699</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>1987</td>
      <td>62.351</td>
      <td>9216418</td>
      <td>706.157306</td>
    </tr>
    <tr>
      <th>1700</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>1992</td>
      <td>60.377</td>
      <td>10704340</td>
      <td>693.420786</td>
    </tr>
    <tr>
      <th>1701</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>1997</td>
      <td>46.809</td>
      <td>11404948</td>
      <td>792.449960</td>
    </tr>
    <tr>
      <th>1702</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>2002</td>
      <td>39.989</td>
      <td>11926563</td>
      <td>672.038623</td>
    </tr>
    <tr>
      <th>1703</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>2007</td>
      <td>43.487</td>
      <td>12311143</td>
      <td>469.709298</td>
    </tr>
  </tbody>
</table>
</div>




```python
# use bitwise operators
# & for and
# | for or
# use ( ) around each condition, this is very important

df.loc[(df['country'] == "Zimbabwe") & (df['year'] > 1990), :]
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>country</th>
      <th>continent</th>
      <th>year</th>
      <th>lifeExp</th>
      <th>pop</th>
      <th>gdpPercap</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1700</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>1992</td>
      <td>60.377</td>
      <td>10704340</td>
      <td>693.420786</td>
    </tr>
    <tr>
      <th>1701</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>1997</td>
      <td>46.809</td>
      <td>11404948</td>
      <td>792.449960</td>
    </tr>
    <tr>
      <th>1702</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>2002</td>
      <td>39.989</td>
      <td>11926563</td>
      <td>672.038623</td>
    </tr>
    <tr>
      <th>1703</th>
      <td>Zimbabwe</td>
      <td>Africa</td>
      <td>2007</td>
      <td>43.487</td>
      <td>12311143</td>
      <td>469.709298</td>
    </tr>
  </tbody>
</table>
</div>




```python

# for each value of year, get the lifeExp column and calculate the mean
# aka split by year, apply the mean function to lifeExp, and combine the results
df.groupby('year')['lifeExp'].mean()
```




    year
    1952    49.057620
    1957    51.507401
    1962    53.609249
    1967    55.678290
    1972    57.647386
    1977    59.570157
    1982    61.533197
    1987    63.212613
    1992    64.160338
    1997    65.014676
    2002    65.694923
    2007    67.007423
    Name: lifeExp, dtype: float64




```python
import numpy as np
```


```python
# use agg or aggregate to pass in your own function
df.groupby("year")["lifeExp"].agg(np.mean)
```




    year
    1952    49.057620
    1957    51.507401
    1962    53.609249
    1967    55.678290
    1972    57.647386
    1977    59.570157
    1982    61.533197
    1987    63.212613
    1992    64.160338
    1997    65.014676
    2002    65.694923
    2007    67.007423
    Name: lifeExp, dtype: float64




```python
# a groupby object doesn't acutlly do anything until you make a calculation
type(df.groupby("year"))
```




    pandas.core.groupby.generic.DataFrameGroupBy




```python
type(df.groupby("year")["lifeExp"])
```




    pandas.core.groupby.generic.SeriesGroupBy




```python
# Calculate multiple descriptive statistics
df.groupby(["year", "continent"])['lifeExp', 'pop', 'gdpPercap'].agg([np.mean, np.std])
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:2: FutureWarning: Indexing with multiple keys (implicitly converted to a tuple of keys) will be deprecated, use a list instead.
      





<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="2" halign="left">lifeExp</th>
      <th colspan="2" halign="left">pop</th>
      <th colspan="2" halign="left">gdpPercap</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
    </tr>
    <tr>
      <th>year</th>
      <th>continent</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">1952</th>
      <th>Africa</th>
      <td>39.135500</td>
      <td>5.151581</td>
      <td>4.570010e+06</td>
      <td>6.317450e+06</td>
      <td>1252.572466</td>
      <td>982.952116</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>53.279840</td>
      <td>9.326082</td>
      <td>1.380610e+07</td>
      <td>3.234163e+07</td>
      <td>4079.062552</td>
      <td>3001.727522</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>46.314394</td>
      <td>9.291751</td>
      <td>4.228356e+07</td>
      <td>1.132267e+08</td>
      <td>5195.484004</td>
      <td>18634.890865</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>64.408500</td>
      <td>6.361088</td>
      <td>1.393736e+07</td>
      <td>1.724745e+07</td>
      <td>5661.057435</td>
      <td>3114.060493</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>69.255000</td>
      <td>0.190919</td>
      <td>5.343003e+06</td>
      <td>4.735083e+06</td>
      <td>10298.085650</td>
      <td>365.560078</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1957</th>
      <th>Africa</th>
      <td>41.266346</td>
      <td>5.620123</td>
      <td>5.093033e+06</td>
      <td>7.076042e+06</td>
      <td>1385.236062</td>
      <td>1134.508918</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>55.960280</td>
      <td>9.033192</td>
      <td>1.547816e+07</td>
      <td>3.553706e+07</td>
      <td>4616.043733</td>
      <td>3312.381083</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>49.318544</td>
      <td>9.635429</td>
      <td>4.735699e+07</td>
      <td>1.280961e+08</td>
      <td>5787.732940</td>
      <td>19506.515959</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>66.703067</td>
      <td>5.295805</td>
      <td>1.459635e+07</td>
      <td>1.783235e+07</td>
      <td>6963.012816</td>
      <td>3677.950146</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>70.295000</td>
      <td>0.049497</td>
      <td>5.970988e+06</td>
      <td>5.291395e+06</td>
      <td>11598.522455</td>
      <td>917.644806</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1962</th>
      <th>Africa</th>
      <td>43.319442</td>
      <td>5.875364</td>
      <td>5.702247e+06</td>
      <td>7.957545e+06</td>
      <td>1598.078825</td>
      <td>1461.839189</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>58.398760</td>
      <td>8.503544</td>
      <td>1.733081e+07</td>
      <td>3.887683e+07</td>
      <td>4901.541870</td>
      <td>3421.740569</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>51.563223</td>
      <td>9.820632</td>
      <td>5.140476e+07</td>
      <td>1.361027e+08</td>
      <td>5729.369625</td>
      <td>16415.857196</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>68.539233</td>
      <td>4.302500</td>
      <td>1.534517e+07</td>
      <td>1.865642e+07</td>
      <td>8365.486814</td>
      <td>4199.193906</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>71.085000</td>
      <td>0.219203</td>
      <td>6.641759e+06</td>
      <td>5.873524e+06</td>
      <td>12696.452430</td>
      <td>677.727301</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1967</th>
      <th>Africa</th>
      <td>45.334538</td>
      <td>6.082673</td>
      <td>6.447875e+06</td>
      <td>8.985505e+06</td>
      <td>2050.363801</td>
      <td>2847.717603</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>60.410920</td>
      <td>7.909171</td>
      <td>1.922986e+07</td>
      <td>4.192559e+07</td>
      <td>5668.253496</td>
      <td>4160.885560</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>54.663640</td>
      <td>9.650965</td>
      <td>5.774736e+07</td>
      <td>1.533418e+08</td>
      <td>5971.173374</td>
      <td>14062.591362</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>69.737600</td>
      <td>3.799728</td>
      <td>1.603930e+07</td>
      <td>1.944359e+07</td>
      <td>10143.823757</td>
      <td>4724.983889</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>71.310000</td>
      <td>0.296985</td>
      <td>7.300207e+06</td>
      <td>6.465865e+06</td>
      <td>14495.021790</td>
      <td>43.986086</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1972</th>
      <th>Africa</th>
      <td>47.450942</td>
      <td>6.416258</td>
      <td>7.305376e+06</td>
      <td>1.013083e+07</td>
      <td>2339.615674</td>
      <td>3286.853884</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>62.394920</td>
      <td>7.323017</td>
      <td>2.117537e+07</td>
      <td>4.493546e+07</td>
      <td>6491.334139</td>
      <td>4754.404329</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>57.319269</td>
      <td>9.722700</td>
      <td>6.518098e+07</td>
      <td>1.740949e+08</td>
      <td>8187.468699</td>
      <td>19087.502918</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>70.775033</td>
      <td>3.240576</td>
      <td>1.668784e+07</td>
      <td>2.018034e+07</td>
      <td>12479.575246</td>
      <td>5509.691411</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>71.910000</td>
      <td>0.028284</td>
      <td>8.053050e+06</td>
      <td>7.246360e+06</td>
      <td>16417.333380</td>
      <td>525.091980</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1977</th>
      <th>Africa</th>
      <td>49.580423</td>
      <td>6.808197</td>
      <td>8.328097e+06</td>
      <td>1.158518e+07</td>
      <td>2585.938508</td>
      <td>4142.398707</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>64.391560</td>
      <td>7.069496</td>
      <td>2.312271e+07</td>
      <td>4.790406e+07</td>
      <td>7352.007126</td>
      <td>5355.602518</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>59.610556</td>
      <td>10.022197</td>
      <td>7.225799e+07</td>
      <td>1.917074e+08</td>
      <td>7791.314020</td>
      <td>11815.777923</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>71.937767</td>
      <td>3.121030</td>
      <td>1.723882e+07</td>
      <td>2.056054e+07</td>
      <td>14283.979110</td>
      <td>5874.464896</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>72.855000</td>
      <td>0.898026</td>
      <td>8.619500e+06</td>
      <td>7.713969e+06</td>
      <td>17283.957605</td>
      <td>1485.263517</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1982</th>
      <th>Africa</th>
      <td>51.592865</td>
      <td>7.375940</td>
      <td>9.602857e+06</td>
      <td>1.345624e+07</td>
      <td>2481.592960</td>
      <td>3242.632753</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>66.228840</td>
      <td>6.720834</td>
      <td>2.521164e+07</td>
      <td>5.129438e+07</td>
      <td>7506.737088</td>
      <td>5530.490471</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>62.617939</td>
      <td>8.535221</td>
      <td>7.909502e+07</td>
      <td>2.065415e+08</td>
      <td>7434.135157</td>
      <td>8701.176499</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>72.806400</td>
      <td>3.218260</td>
      <td>1.770890e+07</td>
      <td>2.097129e+07</td>
      <td>15617.896551</td>
      <td>6453.234827</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>74.290000</td>
      <td>0.636396</td>
      <td>9.197425e+06</td>
      <td>8.466578e+06</td>
      <td>18554.709840</td>
      <td>1304.328377</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1987</th>
      <th>Africa</th>
      <td>53.344788</td>
      <td>7.864089</td>
      <td>1.105450e+07</td>
      <td>1.527748e+07</td>
      <td>2282.668991</td>
      <td>2566.531947</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>68.090720</td>
      <td>5.801929</td>
      <td>2.731016e+07</td>
      <td>5.445969e+07</td>
      <td>7793.400261</td>
      <td>6665.039509</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>64.851182</td>
      <td>8.203792</td>
      <td>8.700669e+07</td>
      <td>2.257332e+08</td>
      <td>7608.226508</td>
      <td>8090.262765</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>73.642167</td>
      <td>3.169680</td>
      <td>1.810314e+07</td>
      <td>2.136971e+07</td>
      <td>17214.310727</td>
      <td>7482.957960</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>75.320000</td>
      <td>1.414214</td>
      <td>9.787208e+06</td>
      <td>9.150020e+06</td>
      <td>20448.040160</td>
      <td>2037.668013</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1992</th>
      <th>Africa</th>
      <td>53.629577</td>
      <td>9.461071</td>
      <td>1.267464e+07</td>
      <td>1.756272e+07</td>
      <td>2281.810333</td>
      <td>2644.075602</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>69.568360</td>
      <td>5.167104</td>
      <td>2.957096e+07</td>
      <td>5.810922e+07</td>
      <td>8044.934406</td>
      <td>7047.089191</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>66.537212</td>
      <td>8.075549</td>
      <td>9.494825e+07</td>
      <td>2.449604e+08</td>
      <td>8639.690248</td>
      <td>9727.431088</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>74.440100</td>
      <td>3.209781</td>
      <td>1.860476e+07</td>
      <td>2.212674e+07</td>
      <td>17061.568084</td>
      <td>9109.804361</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>76.945000</td>
      <td>0.869741</td>
      <td>1.045983e+07</td>
      <td>9.930822e+06</td>
      <td>20894.045885</td>
      <td>3578.979883</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1997</th>
      <th>Africa</th>
      <td>53.598269</td>
      <td>9.103387</td>
      <td>1.430448e+07</td>
      <td>1.987301e+07</td>
      <td>2378.759555</td>
      <td>2820.728117</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>71.150480</td>
      <td>4.887584</td>
      <td>3.187602e+07</td>
      <td>6.203282e+07</td>
      <td>8889.300863</td>
      <td>7874.225145</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>68.020515</td>
      <td>8.091171</td>
      <td>1.025238e+08</td>
      <td>2.623497e+08</td>
      <td>9834.093295</td>
      <td>11094.180481</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>75.505167</td>
      <td>3.104677</td>
      <td>1.896480e+07</td>
      <td>2.274815e+07</td>
      <td>19076.781802</td>
      <td>10065.457716</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>78.190000</td>
      <td>0.905097</td>
      <td>1.112072e+07</td>
      <td>1.052815e+07</td>
      <td>24024.175170</td>
      <td>4205.533703</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">2002</th>
      <th>Africa</th>
      <td>53.325231</td>
      <td>9.586496</td>
      <td>1.603315e+07</td>
      <td>2.230300e+07</td>
      <td>2599.385159</td>
      <td>2972.651308</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>72.422040</td>
      <td>4.799705</td>
      <td>3.399091e+07</td>
      <td>6.560155e+07</td>
      <td>9287.677107</td>
      <td>8895.817785</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>69.233879</td>
      <td>8.374595</td>
      <td>1.091455e+08</td>
      <td>2.767017e+08</td>
      <td>10174.090397</td>
      <td>11150.719203</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>76.700600</td>
      <td>2.922180</td>
      <td>1.927413e+07</td>
      <td>2.322369e+07</td>
      <td>21711.732422</td>
      <td>11197.355517</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>79.740000</td>
      <td>0.890955</td>
      <td>1.172741e+07</td>
      <td>1.105827e+07</td>
      <td>26938.778040</td>
      <td>5301.853680</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">2007</th>
      <th>Africa</th>
      <td>54.806038</td>
      <td>9.630781</td>
      <td>1.787576e+07</td>
      <td>2.491773e+07</td>
      <td>3089.032605</td>
      <td>3618.163491</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>73.608120</td>
      <td>4.440948</td>
      <td>3.595485e+07</td>
      <td>6.883378e+07</td>
      <td>11003.031625</td>
      <td>9713.209302</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>70.728485</td>
      <td>7.963724</td>
      <td>1.155138e+08</td>
      <td>2.896734e+08</td>
      <td>12473.026870</td>
      <td>14154.937343</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>77.648600</td>
      <td>2.979813</td>
      <td>1.953662e+07</td>
      <td>2.362474e+07</td>
      <td>25054.481636</td>
      <td>11800.339811</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>80.719500</td>
      <td>0.729027</td>
      <td>1.227497e+07</td>
      <td>1.153885e+07</td>
      <td>29810.188275</td>
      <td>6540.991104</td>
    </tr>
  </tbody>
</table>
</div>




```python
# split code in different line to make it more readable
df.groupby(
    ["year", "continent"]
    )[
      'lifeExp', 
      'pop', 
      'gdpPercap'
      ].agg([np.mean, np.std])
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:5: FutureWarning: Indexing with multiple keys (implicitly converted to a tuple of keys) will be deprecated, use a list instead.
      """





<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th></th>
      <th colspan="2" halign="left">lifeExp</th>
      <th colspan="2" halign="left">pop</th>
      <th colspan="2" halign="left">gdpPercap</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
    </tr>
    <tr>
      <th>year</th>
      <th>continent</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th rowspan="5" valign="top">1952</th>
      <th>Africa</th>
      <td>39.135500</td>
      <td>5.151581</td>
      <td>4.570010e+06</td>
      <td>6.317450e+06</td>
      <td>1252.572466</td>
      <td>982.952116</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>53.279840</td>
      <td>9.326082</td>
      <td>1.380610e+07</td>
      <td>3.234163e+07</td>
      <td>4079.062552</td>
      <td>3001.727522</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>46.314394</td>
      <td>9.291751</td>
      <td>4.228356e+07</td>
      <td>1.132267e+08</td>
      <td>5195.484004</td>
      <td>18634.890865</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>64.408500</td>
      <td>6.361088</td>
      <td>1.393736e+07</td>
      <td>1.724745e+07</td>
      <td>5661.057435</td>
      <td>3114.060493</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>69.255000</td>
      <td>0.190919</td>
      <td>5.343003e+06</td>
      <td>4.735083e+06</td>
      <td>10298.085650</td>
      <td>365.560078</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1957</th>
      <th>Africa</th>
      <td>41.266346</td>
      <td>5.620123</td>
      <td>5.093033e+06</td>
      <td>7.076042e+06</td>
      <td>1385.236062</td>
      <td>1134.508918</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>55.960280</td>
      <td>9.033192</td>
      <td>1.547816e+07</td>
      <td>3.553706e+07</td>
      <td>4616.043733</td>
      <td>3312.381083</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>49.318544</td>
      <td>9.635429</td>
      <td>4.735699e+07</td>
      <td>1.280961e+08</td>
      <td>5787.732940</td>
      <td>19506.515959</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>66.703067</td>
      <td>5.295805</td>
      <td>1.459635e+07</td>
      <td>1.783235e+07</td>
      <td>6963.012816</td>
      <td>3677.950146</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>70.295000</td>
      <td>0.049497</td>
      <td>5.970988e+06</td>
      <td>5.291395e+06</td>
      <td>11598.522455</td>
      <td>917.644806</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1962</th>
      <th>Africa</th>
      <td>43.319442</td>
      <td>5.875364</td>
      <td>5.702247e+06</td>
      <td>7.957545e+06</td>
      <td>1598.078825</td>
      <td>1461.839189</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>58.398760</td>
      <td>8.503544</td>
      <td>1.733081e+07</td>
      <td>3.887683e+07</td>
      <td>4901.541870</td>
      <td>3421.740569</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>51.563223</td>
      <td>9.820632</td>
      <td>5.140476e+07</td>
      <td>1.361027e+08</td>
      <td>5729.369625</td>
      <td>16415.857196</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>68.539233</td>
      <td>4.302500</td>
      <td>1.534517e+07</td>
      <td>1.865642e+07</td>
      <td>8365.486814</td>
      <td>4199.193906</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>71.085000</td>
      <td>0.219203</td>
      <td>6.641759e+06</td>
      <td>5.873524e+06</td>
      <td>12696.452430</td>
      <td>677.727301</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1967</th>
      <th>Africa</th>
      <td>45.334538</td>
      <td>6.082673</td>
      <td>6.447875e+06</td>
      <td>8.985505e+06</td>
      <td>2050.363801</td>
      <td>2847.717603</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>60.410920</td>
      <td>7.909171</td>
      <td>1.922986e+07</td>
      <td>4.192559e+07</td>
      <td>5668.253496</td>
      <td>4160.885560</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>54.663640</td>
      <td>9.650965</td>
      <td>5.774736e+07</td>
      <td>1.533418e+08</td>
      <td>5971.173374</td>
      <td>14062.591362</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>69.737600</td>
      <td>3.799728</td>
      <td>1.603930e+07</td>
      <td>1.944359e+07</td>
      <td>10143.823757</td>
      <td>4724.983889</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>71.310000</td>
      <td>0.296985</td>
      <td>7.300207e+06</td>
      <td>6.465865e+06</td>
      <td>14495.021790</td>
      <td>43.986086</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1972</th>
      <th>Africa</th>
      <td>47.450942</td>
      <td>6.416258</td>
      <td>7.305376e+06</td>
      <td>1.013083e+07</td>
      <td>2339.615674</td>
      <td>3286.853884</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>62.394920</td>
      <td>7.323017</td>
      <td>2.117537e+07</td>
      <td>4.493546e+07</td>
      <td>6491.334139</td>
      <td>4754.404329</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>57.319269</td>
      <td>9.722700</td>
      <td>6.518098e+07</td>
      <td>1.740949e+08</td>
      <td>8187.468699</td>
      <td>19087.502918</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>70.775033</td>
      <td>3.240576</td>
      <td>1.668784e+07</td>
      <td>2.018034e+07</td>
      <td>12479.575246</td>
      <td>5509.691411</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>71.910000</td>
      <td>0.028284</td>
      <td>8.053050e+06</td>
      <td>7.246360e+06</td>
      <td>16417.333380</td>
      <td>525.091980</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1977</th>
      <th>Africa</th>
      <td>49.580423</td>
      <td>6.808197</td>
      <td>8.328097e+06</td>
      <td>1.158518e+07</td>
      <td>2585.938508</td>
      <td>4142.398707</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>64.391560</td>
      <td>7.069496</td>
      <td>2.312271e+07</td>
      <td>4.790406e+07</td>
      <td>7352.007126</td>
      <td>5355.602518</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>59.610556</td>
      <td>10.022197</td>
      <td>7.225799e+07</td>
      <td>1.917074e+08</td>
      <td>7791.314020</td>
      <td>11815.777923</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>71.937767</td>
      <td>3.121030</td>
      <td>1.723882e+07</td>
      <td>2.056054e+07</td>
      <td>14283.979110</td>
      <td>5874.464896</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>72.855000</td>
      <td>0.898026</td>
      <td>8.619500e+06</td>
      <td>7.713969e+06</td>
      <td>17283.957605</td>
      <td>1485.263517</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1982</th>
      <th>Africa</th>
      <td>51.592865</td>
      <td>7.375940</td>
      <td>9.602857e+06</td>
      <td>1.345624e+07</td>
      <td>2481.592960</td>
      <td>3242.632753</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>66.228840</td>
      <td>6.720834</td>
      <td>2.521164e+07</td>
      <td>5.129438e+07</td>
      <td>7506.737088</td>
      <td>5530.490471</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>62.617939</td>
      <td>8.535221</td>
      <td>7.909502e+07</td>
      <td>2.065415e+08</td>
      <td>7434.135157</td>
      <td>8701.176499</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>72.806400</td>
      <td>3.218260</td>
      <td>1.770890e+07</td>
      <td>2.097129e+07</td>
      <td>15617.896551</td>
      <td>6453.234827</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>74.290000</td>
      <td>0.636396</td>
      <td>9.197425e+06</td>
      <td>8.466578e+06</td>
      <td>18554.709840</td>
      <td>1304.328377</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1987</th>
      <th>Africa</th>
      <td>53.344788</td>
      <td>7.864089</td>
      <td>1.105450e+07</td>
      <td>1.527748e+07</td>
      <td>2282.668991</td>
      <td>2566.531947</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>68.090720</td>
      <td>5.801929</td>
      <td>2.731016e+07</td>
      <td>5.445969e+07</td>
      <td>7793.400261</td>
      <td>6665.039509</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>64.851182</td>
      <td>8.203792</td>
      <td>8.700669e+07</td>
      <td>2.257332e+08</td>
      <td>7608.226508</td>
      <td>8090.262765</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>73.642167</td>
      <td>3.169680</td>
      <td>1.810314e+07</td>
      <td>2.136971e+07</td>
      <td>17214.310727</td>
      <td>7482.957960</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>75.320000</td>
      <td>1.414214</td>
      <td>9.787208e+06</td>
      <td>9.150020e+06</td>
      <td>20448.040160</td>
      <td>2037.668013</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1992</th>
      <th>Africa</th>
      <td>53.629577</td>
      <td>9.461071</td>
      <td>1.267464e+07</td>
      <td>1.756272e+07</td>
      <td>2281.810333</td>
      <td>2644.075602</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>69.568360</td>
      <td>5.167104</td>
      <td>2.957096e+07</td>
      <td>5.810922e+07</td>
      <td>8044.934406</td>
      <td>7047.089191</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>66.537212</td>
      <td>8.075549</td>
      <td>9.494825e+07</td>
      <td>2.449604e+08</td>
      <td>8639.690248</td>
      <td>9727.431088</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>74.440100</td>
      <td>3.209781</td>
      <td>1.860476e+07</td>
      <td>2.212674e+07</td>
      <td>17061.568084</td>
      <td>9109.804361</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>76.945000</td>
      <td>0.869741</td>
      <td>1.045983e+07</td>
      <td>9.930822e+06</td>
      <td>20894.045885</td>
      <td>3578.979883</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">1997</th>
      <th>Africa</th>
      <td>53.598269</td>
      <td>9.103387</td>
      <td>1.430448e+07</td>
      <td>1.987301e+07</td>
      <td>2378.759555</td>
      <td>2820.728117</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>71.150480</td>
      <td>4.887584</td>
      <td>3.187602e+07</td>
      <td>6.203282e+07</td>
      <td>8889.300863</td>
      <td>7874.225145</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>68.020515</td>
      <td>8.091171</td>
      <td>1.025238e+08</td>
      <td>2.623497e+08</td>
      <td>9834.093295</td>
      <td>11094.180481</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>75.505167</td>
      <td>3.104677</td>
      <td>1.896480e+07</td>
      <td>2.274815e+07</td>
      <td>19076.781802</td>
      <td>10065.457716</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>78.190000</td>
      <td>0.905097</td>
      <td>1.112072e+07</td>
      <td>1.052815e+07</td>
      <td>24024.175170</td>
      <td>4205.533703</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">2002</th>
      <th>Africa</th>
      <td>53.325231</td>
      <td>9.586496</td>
      <td>1.603315e+07</td>
      <td>2.230300e+07</td>
      <td>2599.385159</td>
      <td>2972.651308</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>72.422040</td>
      <td>4.799705</td>
      <td>3.399091e+07</td>
      <td>6.560155e+07</td>
      <td>9287.677107</td>
      <td>8895.817785</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>69.233879</td>
      <td>8.374595</td>
      <td>1.091455e+08</td>
      <td>2.767017e+08</td>
      <td>10174.090397</td>
      <td>11150.719203</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>76.700600</td>
      <td>2.922180</td>
      <td>1.927413e+07</td>
      <td>2.322369e+07</td>
      <td>21711.732422</td>
      <td>11197.355517</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>79.740000</td>
      <td>0.890955</td>
      <td>1.172741e+07</td>
      <td>1.105827e+07</td>
      <td>26938.778040</td>
      <td>5301.853680</td>
    </tr>
    <tr>
      <th rowspan="5" valign="top">2007</th>
      <th>Africa</th>
      <td>54.806038</td>
      <td>9.630781</td>
      <td>1.787576e+07</td>
      <td>2.491773e+07</td>
      <td>3089.032605</td>
      <td>3618.163491</td>
    </tr>
    <tr>
      <th>Americas</th>
      <td>73.608120</td>
      <td>4.440948</td>
      <td>3.595485e+07</td>
      <td>6.883378e+07</td>
      <td>11003.031625</td>
      <td>9713.209302</td>
    </tr>
    <tr>
      <th>Asia</th>
      <td>70.728485</td>
      <td>7.963724</td>
      <td>1.155138e+08</td>
      <td>2.896734e+08</td>
      <td>12473.026870</td>
      <td>14154.937343</td>
    </tr>
    <tr>
      <th>Europe</th>
      <td>77.648600</td>
      <td>2.979813</td>
      <td>1.953662e+07</td>
      <td>2.362474e+07</td>
      <td>25054.481636</td>
      <td>11800.339811</td>
    </tr>
    <tr>
      <th>Oceania</th>
      <td>80.719500</td>
      <td>0.729027</td>
      <td>1.227497e+07</td>
      <td>1.153885e+07</td>
      <td>29810.188275</td>
      <td>6540.991104</td>
    </tr>
  </tbody>
</table>
</div>




```python
#Use reset_index to flatten
df.groupby(
    ["year", "continent"]
    )[
      'lifeExp', 
      'pop', 
      'gdpPercap'
      ].agg([np.mean, np.std]).reset_index()


```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:5: FutureWarning: Indexing with multiple keys (implicitly converted to a tuple of keys) will be deprecated, use a list instead.
      """





<div>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th>year</th>
      <th>continent</th>
      <th colspan="2" halign="left">lifeExp</th>
      <th colspan="2" halign="left">pop</th>
      <th colspan="2" halign="left">gdpPercap</th>
    </tr>
    <tr>
      <th></th>
      <th></th>
      <th></th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
      <th>mean</th>
      <th>std</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1952</td>
      <td>Africa</td>
      <td>39.135500</td>
      <td>5.151581</td>
      <td>4.570010e+06</td>
      <td>6.317450e+06</td>
      <td>1252.572466</td>
      <td>982.952116</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1952</td>
      <td>Americas</td>
      <td>53.279840</td>
      <td>9.326082</td>
      <td>1.380610e+07</td>
      <td>3.234163e+07</td>
      <td>4079.062552</td>
      <td>3001.727522</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1952</td>
      <td>Asia</td>
      <td>46.314394</td>
      <td>9.291751</td>
      <td>4.228356e+07</td>
      <td>1.132267e+08</td>
      <td>5195.484004</td>
      <td>18634.890865</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1952</td>
      <td>Europe</td>
      <td>64.408500</td>
      <td>6.361088</td>
      <td>1.393736e+07</td>
      <td>1.724745e+07</td>
      <td>5661.057435</td>
      <td>3114.060493</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1952</td>
      <td>Oceania</td>
      <td>69.255000</td>
      <td>0.190919</td>
      <td>5.343003e+06</td>
      <td>4.735083e+06</td>
      <td>10298.085650</td>
      <td>365.560078</td>
    </tr>
    <tr>
      <th>5</th>
      <td>1957</td>
      <td>Africa</td>
      <td>41.266346</td>
      <td>5.620123</td>
      <td>5.093033e+06</td>
      <td>7.076042e+06</td>
      <td>1385.236062</td>
      <td>1134.508918</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1957</td>
      <td>Americas</td>
      <td>55.960280</td>
      <td>9.033192</td>
      <td>1.547816e+07</td>
      <td>3.553706e+07</td>
      <td>4616.043733</td>
      <td>3312.381083</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1957</td>
      <td>Asia</td>
      <td>49.318544</td>
      <td>9.635429</td>
      <td>4.735699e+07</td>
      <td>1.280961e+08</td>
      <td>5787.732940</td>
      <td>19506.515959</td>
    </tr>
    <tr>
      <th>8</th>
      <td>1957</td>
      <td>Europe</td>
      <td>66.703067</td>
      <td>5.295805</td>
      <td>1.459635e+07</td>
      <td>1.783235e+07</td>
      <td>6963.012816</td>
      <td>3677.950146</td>
    </tr>
    <tr>
      <th>9</th>
      <td>1957</td>
      <td>Oceania</td>
      <td>70.295000</td>
      <td>0.049497</td>
      <td>5.970988e+06</td>
      <td>5.291395e+06</td>
      <td>11598.522455</td>
      <td>917.644806</td>
    </tr>
    <tr>
      <th>10</th>
      <td>1962</td>
      <td>Africa</td>
      <td>43.319442</td>
      <td>5.875364</td>
      <td>5.702247e+06</td>
      <td>7.957545e+06</td>
      <td>1598.078825</td>
      <td>1461.839189</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1962</td>
      <td>Americas</td>
      <td>58.398760</td>
      <td>8.503544</td>
      <td>1.733081e+07</td>
      <td>3.887683e+07</td>
      <td>4901.541870</td>
      <td>3421.740569</td>
    </tr>
    <tr>
      <th>12</th>
      <td>1962</td>
      <td>Asia</td>
      <td>51.563223</td>
      <td>9.820632</td>
      <td>5.140476e+07</td>
      <td>1.361027e+08</td>
      <td>5729.369625</td>
      <td>16415.857196</td>
    </tr>
    <tr>
      <th>13</th>
      <td>1962</td>
      <td>Europe</td>
      <td>68.539233</td>
      <td>4.302500</td>
      <td>1.534517e+07</td>
      <td>1.865642e+07</td>
      <td>8365.486814</td>
      <td>4199.193906</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1962</td>
      <td>Oceania</td>
      <td>71.085000</td>
      <td>0.219203</td>
      <td>6.641759e+06</td>
      <td>5.873524e+06</td>
      <td>12696.452430</td>
      <td>677.727301</td>
    </tr>
    <tr>
      <th>15</th>
      <td>1967</td>
      <td>Africa</td>
      <td>45.334538</td>
      <td>6.082673</td>
      <td>6.447875e+06</td>
      <td>8.985505e+06</td>
      <td>2050.363801</td>
      <td>2847.717603</td>
    </tr>
    <tr>
      <th>16</th>
      <td>1967</td>
      <td>Americas</td>
      <td>60.410920</td>
      <td>7.909171</td>
      <td>1.922986e+07</td>
      <td>4.192559e+07</td>
      <td>5668.253496</td>
      <td>4160.885560</td>
    </tr>
    <tr>
      <th>17</th>
      <td>1967</td>
      <td>Asia</td>
      <td>54.663640</td>
      <td>9.650965</td>
      <td>5.774736e+07</td>
      <td>1.533418e+08</td>
      <td>5971.173374</td>
      <td>14062.591362</td>
    </tr>
    <tr>
      <th>18</th>
      <td>1967</td>
      <td>Europe</td>
      <td>69.737600</td>
      <td>3.799728</td>
      <td>1.603930e+07</td>
      <td>1.944359e+07</td>
      <td>10143.823757</td>
      <td>4724.983889</td>
    </tr>
    <tr>
      <th>19</th>
      <td>1967</td>
      <td>Oceania</td>
      <td>71.310000</td>
      <td>0.296985</td>
      <td>7.300207e+06</td>
      <td>6.465865e+06</td>
      <td>14495.021790</td>
      <td>43.986086</td>
    </tr>
    <tr>
      <th>20</th>
      <td>1972</td>
      <td>Africa</td>
      <td>47.450942</td>
      <td>6.416258</td>
      <td>7.305376e+06</td>
      <td>1.013083e+07</td>
      <td>2339.615674</td>
      <td>3286.853884</td>
    </tr>
    <tr>
      <th>21</th>
      <td>1972</td>
      <td>Americas</td>
      <td>62.394920</td>
      <td>7.323017</td>
      <td>2.117537e+07</td>
      <td>4.493546e+07</td>
      <td>6491.334139</td>
      <td>4754.404329</td>
    </tr>
    <tr>
      <th>22</th>
      <td>1972</td>
      <td>Asia</td>
      <td>57.319269</td>
      <td>9.722700</td>
      <td>6.518098e+07</td>
      <td>1.740949e+08</td>
      <td>8187.468699</td>
      <td>19087.502918</td>
    </tr>
    <tr>
      <th>23</th>
      <td>1972</td>
      <td>Europe</td>
      <td>70.775033</td>
      <td>3.240576</td>
      <td>1.668784e+07</td>
      <td>2.018034e+07</td>
      <td>12479.575246</td>
      <td>5509.691411</td>
    </tr>
    <tr>
      <th>24</th>
      <td>1972</td>
      <td>Oceania</td>
      <td>71.910000</td>
      <td>0.028284</td>
      <td>8.053050e+06</td>
      <td>7.246360e+06</td>
      <td>16417.333380</td>
      <td>525.091980</td>
    </tr>
    <tr>
      <th>25</th>
      <td>1977</td>
      <td>Africa</td>
      <td>49.580423</td>
      <td>6.808197</td>
      <td>8.328097e+06</td>
      <td>1.158518e+07</td>
      <td>2585.938508</td>
      <td>4142.398707</td>
    </tr>
    <tr>
      <th>26</th>
      <td>1977</td>
      <td>Americas</td>
      <td>64.391560</td>
      <td>7.069496</td>
      <td>2.312271e+07</td>
      <td>4.790406e+07</td>
      <td>7352.007126</td>
      <td>5355.602518</td>
    </tr>
    <tr>
      <th>27</th>
      <td>1977</td>
      <td>Asia</td>
      <td>59.610556</td>
      <td>10.022197</td>
      <td>7.225799e+07</td>
      <td>1.917074e+08</td>
      <td>7791.314020</td>
      <td>11815.777923</td>
    </tr>
    <tr>
      <th>28</th>
      <td>1977</td>
      <td>Europe</td>
      <td>71.937767</td>
      <td>3.121030</td>
      <td>1.723882e+07</td>
      <td>2.056054e+07</td>
      <td>14283.979110</td>
      <td>5874.464896</td>
    </tr>
    <tr>
      <th>29</th>
      <td>1977</td>
      <td>Oceania</td>
      <td>72.855000</td>
      <td>0.898026</td>
      <td>8.619500e+06</td>
      <td>7.713969e+06</td>
      <td>17283.957605</td>
      <td>1485.263517</td>
    </tr>
    <tr>
      <th>30</th>
      <td>1982</td>
      <td>Africa</td>
      <td>51.592865</td>
      <td>7.375940</td>
      <td>9.602857e+06</td>
      <td>1.345624e+07</td>
      <td>2481.592960</td>
      <td>3242.632753</td>
    </tr>
    <tr>
      <th>31</th>
      <td>1982</td>
      <td>Americas</td>
      <td>66.228840</td>
      <td>6.720834</td>
      <td>2.521164e+07</td>
      <td>5.129438e+07</td>
      <td>7506.737088</td>
      <td>5530.490471</td>
    </tr>
    <tr>
      <th>32</th>
      <td>1982</td>
      <td>Asia</td>
      <td>62.617939</td>
      <td>8.535221</td>
      <td>7.909502e+07</td>
      <td>2.065415e+08</td>
      <td>7434.135157</td>
      <td>8701.176499</td>
    </tr>
    <tr>
      <th>33</th>
      <td>1982</td>
      <td>Europe</td>
      <td>72.806400</td>
      <td>3.218260</td>
      <td>1.770890e+07</td>
      <td>2.097129e+07</td>
      <td>15617.896551</td>
      <td>6453.234827</td>
    </tr>
    <tr>
      <th>34</th>
      <td>1982</td>
      <td>Oceania</td>
      <td>74.290000</td>
      <td>0.636396</td>
      <td>9.197425e+06</td>
      <td>8.466578e+06</td>
      <td>18554.709840</td>
      <td>1304.328377</td>
    </tr>
    <tr>
      <th>35</th>
      <td>1987</td>
      <td>Africa</td>
      <td>53.344788</td>
      <td>7.864089</td>
      <td>1.105450e+07</td>
      <td>1.527748e+07</td>
      <td>2282.668991</td>
      <td>2566.531947</td>
    </tr>
    <tr>
      <th>36</th>
      <td>1987</td>
      <td>Americas</td>
      <td>68.090720</td>
      <td>5.801929</td>
      <td>2.731016e+07</td>
      <td>5.445969e+07</td>
      <td>7793.400261</td>
      <td>6665.039509</td>
    </tr>
    <tr>
      <th>37</th>
      <td>1987</td>
      <td>Asia</td>
      <td>64.851182</td>
      <td>8.203792</td>
      <td>8.700669e+07</td>
      <td>2.257332e+08</td>
      <td>7608.226508</td>
      <td>8090.262765</td>
    </tr>
    <tr>
      <th>38</th>
      <td>1987</td>
      <td>Europe</td>
      <td>73.642167</td>
      <td>3.169680</td>
      <td>1.810314e+07</td>
      <td>2.136971e+07</td>
      <td>17214.310727</td>
      <td>7482.957960</td>
    </tr>
    <tr>
      <th>39</th>
      <td>1987</td>
      <td>Oceania</td>
      <td>75.320000</td>
      <td>1.414214</td>
      <td>9.787208e+06</td>
      <td>9.150020e+06</td>
      <td>20448.040160</td>
      <td>2037.668013</td>
    </tr>
    <tr>
      <th>40</th>
      <td>1992</td>
      <td>Africa</td>
      <td>53.629577</td>
      <td>9.461071</td>
      <td>1.267464e+07</td>
      <td>1.756272e+07</td>
      <td>2281.810333</td>
      <td>2644.075602</td>
    </tr>
    <tr>
      <th>41</th>
      <td>1992</td>
      <td>Americas</td>
      <td>69.568360</td>
      <td>5.167104</td>
      <td>2.957096e+07</td>
      <td>5.810922e+07</td>
      <td>8044.934406</td>
      <td>7047.089191</td>
    </tr>
    <tr>
      <th>42</th>
      <td>1992</td>
      <td>Asia</td>
      <td>66.537212</td>
      <td>8.075549</td>
      <td>9.494825e+07</td>
      <td>2.449604e+08</td>
      <td>8639.690248</td>
      <td>9727.431088</td>
    </tr>
    <tr>
      <th>43</th>
      <td>1992</td>
      <td>Europe</td>
      <td>74.440100</td>
      <td>3.209781</td>
      <td>1.860476e+07</td>
      <td>2.212674e+07</td>
      <td>17061.568084</td>
      <td>9109.804361</td>
    </tr>
    <tr>
      <th>44</th>
      <td>1992</td>
      <td>Oceania</td>
      <td>76.945000</td>
      <td>0.869741</td>
      <td>1.045983e+07</td>
      <td>9.930822e+06</td>
      <td>20894.045885</td>
      <td>3578.979883</td>
    </tr>
    <tr>
      <th>45</th>
      <td>1997</td>
      <td>Africa</td>
      <td>53.598269</td>
      <td>9.103387</td>
      <td>1.430448e+07</td>
      <td>1.987301e+07</td>
      <td>2378.759555</td>
      <td>2820.728117</td>
    </tr>
    <tr>
      <th>46</th>
      <td>1997</td>
      <td>Americas</td>
      <td>71.150480</td>
      <td>4.887584</td>
      <td>3.187602e+07</td>
      <td>6.203282e+07</td>
      <td>8889.300863</td>
      <td>7874.225145</td>
    </tr>
    <tr>
      <th>47</th>
      <td>1997</td>
      <td>Asia</td>
      <td>68.020515</td>
      <td>8.091171</td>
      <td>1.025238e+08</td>
      <td>2.623497e+08</td>
      <td>9834.093295</td>
      <td>11094.180481</td>
    </tr>
    <tr>
      <th>48</th>
      <td>1997</td>
      <td>Europe</td>
      <td>75.505167</td>
      <td>3.104677</td>
      <td>1.896480e+07</td>
      <td>2.274815e+07</td>
      <td>19076.781802</td>
      <td>10065.457716</td>
    </tr>
    <tr>
      <th>49</th>
      <td>1997</td>
      <td>Oceania</td>
      <td>78.190000</td>
      <td>0.905097</td>
      <td>1.112072e+07</td>
      <td>1.052815e+07</td>
      <td>24024.175170</td>
      <td>4205.533703</td>
    </tr>
    <tr>
      <th>50</th>
      <td>2002</td>
      <td>Africa</td>
      <td>53.325231</td>
      <td>9.586496</td>
      <td>1.603315e+07</td>
      <td>2.230300e+07</td>
      <td>2599.385159</td>
      <td>2972.651308</td>
    </tr>
    <tr>
      <th>51</th>
      <td>2002</td>
      <td>Americas</td>
      <td>72.422040</td>
      <td>4.799705</td>
      <td>3.399091e+07</td>
      <td>6.560155e+07</td>
      <td>9287.677107</td>
      <td>8895.817785</td>
    </tr>
    <tr>
      <th>52</th>
      <td>2002</td>
      <td>Asia</td>
      <td>69.233879</td>
      <td>8.374595</td>
      <td>1.091455e+08</td>
      <td>2.767017e+08</td>
      <td>10174.090397</td>
      <td>11150.719203</td>
    </tr>
    <tr>
      <th>53</th>
      <td>2002</td>
      <td>Europe</td>
      <td>76.700600</td>
      <td>2.922180</td>
      <td>1.927413e+07</td>
      <td>2.322369e+07</td>
      <td>21711.732422</td>
      <td>11197.355517</td>
    </tr>
    <tr>
      <th>54</th>
      <td>2002</td>
      <td>Oceania</td>
      <td>79.740000</td>
      <td>0.890955</td>
      <td>1.172741e+07</td>
      <td>1.105827e+07</td>
      <td>26938.778040</td>
      <td>5301.853680</td>
    </tr>
    <tr>
      <th>55</th>
      <td>2007</td>
      <td>Africa</td>
      <td>54.806038</td>
      <td>9.630781</td>
      <td>1.787576e+07</td>
      <td>2.491773e+07</td>
      <td>3089.032605</td>
      <td>3618.163491</td>
    </tr>
    <tr>
      <th>56</th>
      <td>2007</td>
      <td>Americas</td>
      <td>73.608120</td>
      <td>4.440948</td>
      <td>3.595485e+07</td>
      <td>6.883378e+07</td>
      <td>11003.031625</td>
      <td>9713.209302</td>
    </tr>
    <tr>
      <th>57</th>
      <td>2007</td>
      <td>Asia</td>
      <td>70.728485</td>
      <td>7.963724</td>
      <td>1.155138e+08</td>
      <td>2.896734e+08</td>
      <td>12473.026870</td>
      <td>14154.937343</td>
    </tr>
    <tr>
      <th>58</th>
      <td>2007</td>
      <td>Europe</td>
      <td>77.648600</td>
      <td>2.979813</td>
      <td>1.953662e+07</td>
      <td>2.362474e+07</td>
      <td>25054.481636</td>
      <td>11800.339811</td>
    </tr>
    <tr>
      <th>59</th>
      <td>2007</td>
      <td>Oceania</td>
      <td>80.719500</td>
      <td>0.729027</td>
      <td>1.227497e+07</td>
      <td>1.153885e+07</td>
      <td>29810.188275</td>
      <td>6540.991104</td>
    </tr>
  </tbody>
</table>
</div>



Chapter-2 - Tidy
----


```python
pew = pd.read_csv("https://raw.githubusercontent.com/himnsuk/2021-07-13-scipy-pandas/main/data/pew.csv")
```


```python
pew
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>religion</th>
      <th>&lt;$10k</th>
      <th>$10-20k</th>
      <th>$20-30k</th>
      <th>$30-40k</th>
      <th>$40-50k</th>
      <th>$50-75k</th>
      <th>$75-100k</th>
      <th>$100-150k</th>
      <th>&gt;150k</th>
      <th>Don't know/refused</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Agnostic</td>
      <td>27</td>
      <td>34</td>
      <td>60</td>
      <td>81</td>
      <td>76</td>
      <td>137</td>
      <td>122</td>
      <td>109</td>
      <td>84</td>
      <td>96</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Atheist</td>
      <td>12</td>
      <td>27</td>
      <td>37</td>
      <td>52</td>
      <td>35</td>
      <td>70</td>
      <td>73</td>
      <td>59</td>
      <td>74</td>
      <td>76</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Buddhist</td>
      <td>27</td>
      <td>21</td>
      <td>30</td>
      <td>34</td>
      <td>33</td>
      <td>58</td>
      <td>62</td>
      <td>39</td>
      <td>53</td>
      <td>54</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Catholic</td>
      <td>418</td>
      <td>617</td>
      <td>732</td>
      <td>670</td>
      <td>638</td>
      <td>1116</td>
      <td>949</td>
      <td>792</td>
      <td>633</td>
      <td>1489</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Don’t know/refused</td>
      <td>15</td>
      <td>14</td>
      <td>15</td>
      <td>11</td>
      <td>10</td>
      <td>35</td>
      <td>21</td>
      <td>17</td>
      <td>18</td>
      <td>116</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Evangelical Prot</td>
      <td>575</td>
      <td>869</td>
      <td>1064</td>
      <td>982</td>
      <td>881</td>
      <td>1486</td>
      <td>949</td>
      <td>723</td>
      <td>414</td>
      <td>1529</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Hindu</td>
      <td>1</td>
      <td>9</td>
      <td>7</td>
      <td>9</td>
      <td>11</td>
      <td>34</td>
      <td>47</td>
      <td>48</td>
      <td>54</td>
      <td>37</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Historically Black Prot</td>
      <td>228</td>
      <td>244</td>
      <td>236</td>
      <td>238</td>
      <td>197</td>
      <td>223</td>
      <td>131</td>
      <td>81</td>
      <td>78</td>
      <td>339</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Jehovah's Witness</td>
      <td>20</td>
      <td>27</td>
      <td>24</td>
      <td>24</td>
      <td>21</td>
      <td>30</td>
      <td>15</td>
      <td>11</td>
      <td>6</td>
      <td>37</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Jewish</td>
      <td>19</td>
      <td>19</td>
      <td>25</td>
      <td>25</td>
      <td>30</td>
      <td>95</td>
      <td>69</td>
      <td>87</td>
      <td>151</td>
      <td>162</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Mainline Prot</td>
      <td>289</td>
      <td>495</td>
      <td>619</td>
      <td>655</td>
      <td>651</td>
      <td>1107</td>
      <td>939</td>
      <td>753</td>
      <td>634</td>
      <td>1328</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Mormon</td>
      <td>29</td>
      <td>40</td>
      <td>48</td>
      <td>51</td>
      <td>56</td>
      <td>112</td>
      <td>85</td>
      <td>49</td>
      <td>42</td>
      <td>69</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Muslim</td>
      <td>6</td>
      <td>7</td>
      <td>9</td>
      <td>10</td>
      <td>9</td>
      <td>23</td>
      <td>16</td>
      <td>8</td>
      <td>6</td>
      <td>22</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Orthodox</td>
      <td>13</td>
      <td>17</td>
      <td>23</td>
      <td>32</td>
      <td>32</td>
      <td>47</td>
      <td>38</td>
      <td>42</td>
      <td>46</td>
      <td>73</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Other Christian</td>
      <td>9</td>
      <td>7</td>
      <td>11</td>
      <td>13</td>
      <td>13</td>
      <td>14</td>
      <td>18</td>
      <td>14</td>
      <td>12</td>
      <td>18</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Other Faiths</td>
      <td>20</td>
      <td>33</td>
      <td>40</td>
      <td>46</td>
      <td>49</td>
      <td>63</td>
      <td>46</td>
      <td>40</td>
      <td>41</td>
      <td>71</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Other World Religions</td>
      <td>5</td>
      <td>2</td>
      <td>3</td>
      <td>4</td>
      <td>2</td>
      <td>7</td>
      <td>3</td>
      <td>4</td>
      <td>4</td>
      <td>8</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Unaffiliated</td>
      <td>217</td>
      <td>299</td>
      <td>374</td>
      <td>365</td>
      <td>341</td>
      <td>528</td>
      <td>407</td>
      <td>321</td>
      <td>258</td>
      <td>597</td>
    </tr>
  </tbody>
</table>
</div>




```python
pew.melt(id_vars=["religion"])
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>religion</th>
      <th>variable</th>
      <th>value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Agnostic</td>
      <td>&lt;$10k</td>
      <td>27</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Atheist</td>
      <td>&lt;$10k</td>
      <td>12</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Buddhist</td>
      <td>&lt;$10k</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Catholic</td>
      <td>&lt;$10k</td>
      <td>418</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Don’t know/refused</td>
      <td>&lt;$10k</td>
      <td>15</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>175</th>
      <td>Orthodox</td>
      <td>Don't know/refused</td>
      <td>73</td>
    </tr>
    <tr>
      <th>176</th>
      <td>Other Christian</td>
      <td>Don't know/refused</td>
      <td>18</td>
    </tr>
    <tr>
      <th>177</th>
      <td>Other Faiths</td>
      <td>Don't know/refused</td>
      <td>71</td>
    </tr>
    <tr>
      <th>178</th>
      <td>Other World Religions</td>
      <td>Don't know/refused</td>
      <td>8</td>
    </tr>
    <tr>
      <th>179</th>
      <td>Unaffiliated</td>
      <td>Don't know/refused</td>
      <td>597</td>
    </tr>
  </tbody>
</table>
<p>180 rows × 3 columns</p>
</div>




```python
pew_tidy = pew.melt(id_vars=['religion'],var_name="income", value_name="count")
```


```python
pew_tidy.groupby('income')['count'].sum()
```




    income
    $10-20k               2781
    $100-150k             3197
    $20-30k               3357
    $30-40k               3302
    $40-50k               3085
    $50-75k               5185
    $75-100k              3990
    <$10k                 1930
    >150k                 2608
    Don't know/refused    6121
    Name: count, dtype: int64




```python
pew_tidy.groupby('income')['count'].agg(['sum', 'mean', 'std', 'var', 'min', 'max']).reset_index()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>income</th>
      <th>sum</th>
      <th>mean</th>
      <th>std</th>
      <th>var</th>
      <th>min</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>$10-20k</td>
      <td>2781</td>
      <td>154.500000</td>
      <td>255.172433</td>
      <td>65112.970588</td>
      <td>2</td>
      <td>869</td>
    </tr>
    <tr>
      <th>1</th>
      <td>$100-150k</td>
      <td>3197</td>
      <td>177.611111</td>
      <td>275.679724</td>
      <td>75999.310458</td>
      <td>4</td>
      <td>792</td>
    </tr>
    <tr>
      <th>2</th>
      <td>$20-30k</td>
      <td>3357</td>
      <td>186.500000</td>
      <td>309.891869</td>
      <td>96032.970588</td>
      <td>3</td>
      <td>1064</td>
    </tr>
    <tr>
      <th>3</th>
      <td>$30-40k</td>
      <td>3302</td>
      <td>183.444444</td>
      <td>291.470354</td>
      <td>84954.967320</td>
      <td>4</td>
      <td>982</td>
    </tr>
    <tr>
      <th>4</th>
      <td>$40-50k</td>
      <td>3085</td>
      <td>171.388889</td>
      <td>271.144446</td>
      <td>73519.310458</td>
      <td>2</td>
      <td>881</td>
    </tr>
    <tr>
      <th>5</th>
      <td>$50-75k</td>
      <td>5185</td>
      <td>288.055556</td>
      <td>458.442436</td>
      <td>210169.467320</td>
      <td>7</td>
      <td>1486</td>
    </tr>
    <tr>
      <th>6</th>
      <td>$75-100k</td>
      <td>3990</td>
      <td>221.666667</td>
      <td>345.078849</td>
      <td>119079.411765</td>
      <td>3</td>
      <td>949</td>
    </tr>
    <tr>
      <th>7</th>
      <td>&lt;$10k</td>
      <td>1930</td>
      <td>107.222222</td>
      <td>168.931784</td>
      <td>28537.947712</td>
      <td>1</td>
      <td>575</td>
    </tr>
    <tr>
      <th>8</th>
      <td>&gt;150k</td>
      <td>2608</td>
      <td>144.888889</td>
      <td>205.224952</td>
      <td>42117.281046</td>
      <td>4</td>
      <td>634</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Don't know/refused</td>
      <td>6121</td>
      <td>340.055556</td>
      <td>530.523878</td>
      <td>281455.584967</td>
      <td>8</td>
      <td>1529</td>
    </tr>
  </tbody>
</table>
</div>




```python
billboard = pd.read_csv("https://raw.githubusercontent.com/himnsuk/2021-07-13-scipy-pandas/main/data/billboard.csv")
```


```python
billboard.head()
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>artist</th>
      <th>track</th>
      <th>time</th>
      <th>date.entered</th>
      <th>wk1</th>
      <th>wk2</th>
      <th>wk3</th>
      <th>wk4</th>
      <th>wk5</th>
      <th>wk6</th>
      <th>wk7</th>
      <th>wk8</th>
      <th>wk9</th>
      <th>wk10</th>
      <th>wk11</th>
      <th>wk12</th>
      <th>wk13</th>
      <th>wk14</th>
      <th>wk15</th>
      <th>wk16</th>
      <th>wk17</th>
      <th>wk18</th>
      <th>wk19</th>
      <th>wk20</th>
      <th>wk21</th>
      <th>wk22</th>
      <th>wk23</th>
      <th>wk24</th>
      <th>wk25</th>
      <th>wk26</th>
      <th>wk27</th>
      <th>wk28</th>
      <th>wk29</th>
      <th>wk30</th>
      <th>wk31</th>
      <th>wk32</th>
      <th>wk33</th>
      <th>wk34</th>
      <th>wk35</th>
      <th>...</th>
      <th>wk37</th>
      <th>wk38</th>
      <th>wk39</th>
      <th>wk40</th>
      <th>wk41</th>
      <th>wk42</th>
      <th>wk43</th>
      <th>wk44</th>
      <th>wk45</th>
      <th>wk46</th>
      <th>wk47</th>
      <th>wk48</th>
      <th>wk49</th>
      <th>wk50</th>
      <th>wk51</th>
      <th>wk52</th>
      <th>wk53</th>
      <th>wk54</th>
      <th>wk55</th>
      <th>wk56</th>
      <th>wk57</th>
      <th>wk58</th>
      <th>wk59</th>
      <th>wk60</th>
      <th>wk61</th>
      <th>wk62</th>
      <th>wk63</th>
      <th>wk64</th>
      <th>wk65</th>
      <th>wk66</th>
      <th>wk67</th>
      <th>wk68</th>
      <th>wk69</th>
      <th>wk70</th>
      <th>wk71</th>
      <th>wk72</th>
      <th>wk73</th>
      <th>wk74</th>
      <th>wk75</th>
      <th>wk76</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2000</td>
      <td>2 Pac</td>
      <td>Baby Don't Cry (Keep...</td>
      <td>4:22</td>
      <td>2000-02-26</td>
      <td>87</td>
      <td>82.0</td>
      <td>72.0</td>
      <td>77.0</td>
      <td>87.0</td>
      <td>94.0</td>
      <td>99.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2000</td>
      <td>2Ge+her</td>
      <td>The Hardest Part Of ...</td>
      <td>3:15</td>
      <td>2000-09-02</td>
      <td>91</td>
      <td>87.0</td>
      <td>92.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2000</td>
      <td>3 Doors Down</td>
      <td>Kryptonite</td>
      <td>3:53</td>
      <td>2000-04-08</td>
      <td>81</td>
      <td>70.0</td>
      <td>68.0</td>
      <td>67.0</td>
      <td>66.0</td>
      <td>57.0</td>
      <td>54.0</td>
      <td>53.0</td>
      <td>51.0</td>
      <td>51.0</td>
      <td>51.0</td>
      <td>51.0</td>
      <td>47.0</td>
      <td>44.0</td>
      <td>38.0</td>
      <td>28.0</td>
      <td>22.0</td>
      <td>18.0</td>
      <td>18.0</td>
      <td>14.0</td>
      <td>12.0</td>
      <td>7.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>6.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>4.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>5.0</td>
      <td>9.0</td>
      <td>9.0</td>
      <td>15.0</td>
      <td>14.0</td>
      <td>13.0</td>
      <td>14.0</td>
      <td>16.0</td>
      <td>17.0</td>
      <td>21.0</td>
      <td>22.0</td>
      <td>24.0</td>
      <td>28.0</td>
      <td>33.0</td>
      <td>42.0</td>
      <td>42.0</td>
      <td>49.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2000</td>
      <td>3 Doors Down</td>
      <td>Loser</td>
      <td>4:24</td>
      <td>2000-10-21</td>
      <td>76</td>
      <td>76.0</td>
      <td>72.0</td>
      <td>69.0</td>
      <td>67.0</td>
      <td>65.0</td>
      <td>55.0</td>
      <td>59.0</td>
      <td>62.0</td>
      <td>61.0</td>
      <td>61.0</td>
      <td>59.0</td>
      <td>61.0</td>
      <td>66.0</td>
      <td>72.0</td>
      <td>76.0</td>
      <td>75.0</td>
      <td>67.0</td>
      <td>73.0</td>
      <td>70.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2000</td>
      <td>504 Boyz</td>
      <td>Wobble Wobble</td>
      <td>3:35</td>
      <td>2000-04-15</td>
      <td>57</td>
      <td>34.0</td>
      <td>25.0</td>
      <td>17.0</td>
      <td>17.0</td>
      <td>31.0</td>
      <td>36.0</td>
      <td>49.0</td>
      <td>53.0</td>
      <td>57.0</td>
      <td>64.0</td>
      <td>70.0</td>
      <td>75.0</td>
      <td>76.0</td>
      <td>78.0</td>
      <td>85.0</td>
      <td>92.0</td>
      <td>96.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 81 columns</p>
</div>




```python
# use [ ] if you need to pass in multiple columns into id_vars

billboard.melt(id_vars=['year', 'artist', 'track', 'time', 'date.entered'],
               value_name="rank",
               var_name="week")
```




<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>year</th>
      <th>artist</th>
      <th>track</th>
      <th>time</th>
      <th>date.entered</th>
      <th>week</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2000</td>
      <td>2 Pac</td>
      <td>Baby Don't Cry (Keep...</td>
      <td>4:22</td>
      <td>2000-02-26</td>
      <td>wk1</td>
      <td>87.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2000</td>
      <td>2Ge+her</td>
      <td>The Hardest Part Of ...</td>
      <td>3:15</td>
      <td>2000-09-02</td>
      <td>wk1</td>
      <td>91.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2000</td>
      <td>3 Doors Down</td>
      <td>Kryptonite</td>
      <td>3:53</td>
      <td>2000-04-08</td>
      <td>wk1</td>
      <td>81.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2000</td>
      <td>3 Doors Down</td>
      <td>Loser</td>
      <td>4:24</td>
      <td>2000-10-21</td>
      <td>wk1</td>
      <td>76.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2000</td>
      <td>504 Boyz</td>
      <td>Wobble Wobble</td>
      <td>3:35</td>
      <td>2000-04-15</td>
      <td>wk1</td>
      <td>57.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>24087</th>
      <td>2000</td>
      <td>Yankee Grey</td>
      <td>Another Nine Minutes</td>
      <td>3:10</td>
      <td>2000-04-29</td>
      <td>wk76</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>24088</th>
      <td>2000</td>
      <td>Yearwood, Trisha</td>
      <td>Real Live Woman</td>
      <td>3:55</td>
      <td>2000-04-01</td>
      <td>wk76</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>24089</th>
      <td>2000</td>
      <td>Ying Yang Twins</td>
      <td>Whistle While You Tw...</td>
      <td>4:19</td>
      <td>2000-03-18</td>
      <td>wk76</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>24090</th>
      <td>2000</td>
      <td>Zombie Nation</td>
      <td>Kernkraft 400</td>
      <td>3:30</td>
      <td>2000-09-02</td>
      <td>wk76</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>24091</th>
      <td>2000</td>
      <td>matchbox twenty</td>
      <td>Bent</td>
      <td>4:12</td>
      <td>2000-04-29</td>
      <td>wk76</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>24092 rows × 7 columns</p>
</div>