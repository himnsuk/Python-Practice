**Encoding Techniques in Data Science**

### 1. **OneHot Encoding**
**Definition:**
OneHot Encoding converts categorical variables into binary vectors where each category is represented by a separate column containing 0s and 1s.

**Use Cases:**
- Suitable for nominal categorical variables (no order or ranking).
- Commonly used in classification models requiring categorical input.

**Example:**
| Color |
|--------|
| Red     |
| Blue    |
| Green   |

After OneHot Encoding:
| Color_Red | Color_Blue | Color_Green |
|------------|-------------|---------------|
| 1           | 0           | 0             |
| 0           | 1           | 0             |
| 0           | 0           | 1             |

**Mathematics:**
- No complex formula; maps each category to a unique vector.

**Advantages:**
- Prevents the model from assuming any ordinal relationship.
- Widely supported in ML libraries.

**Disadvantages:**
- High cardinality features can create excessive dimensions, leading to increased model complexity.

---

### 2. **Label Encoding**
**Definition:**
Label Encoding assigns a unique integer value to each category in the feature.

**Use Cases:**
- Suitable for ordinal categorical variables (where order matters).
- Useful in tree-based models which can interpret numerical relationships better.

**Example:**
| Size |
|------|
| Small |
| Medium|
| Large |

After Label Encoding:
| Size |
|------|
| 0    |
| 1    |
| 2    |

**Mathematics:**
- Maps each unique category to an integer value: \( X_i = i \)

**Advantages:**
- Efficient for low-cardinality categorical features.

**Disadvantages:**
- May introduce an unintended ordinal relationship in non-ordinal data.

---

### 3. **Ordinal Encoding**
**Definition:**
Ordinal Encoding assigns integer values to categories based on their ordinal rank.

**Use Cases:**
- Ideal for ordinal data where category order carries meaning.

**Example:**
| Education |
|------------|
| High School |
| Bachelor's  |
| Master's     |
| PhD          |

After Ordinal Encoding:
| Education |
|------------|
| 1          |
| 2          |
| 3          |
| 4          |

**Mathematics:**
- Assigns ranks to categories, often starting from 1: \( X_i = i \)

**Advantages:**
- Preserves meaningful ordinal relationships.

**Disadvantages:**
- Misinterpreting non-ordinal data as ordinal can mislead the model.

---

### 4. **Target Encoding**
**Definition:**
Target Encoding replaces each category with the mean of the target variable for that category.

**Use Cases:**
- Effective for high-cardinality categorical features in regression problems.

**Example:**
| City  | House Price |
|-------|-------------|
| A      | 300,000     |
| B      | 500,000     |
| A      | 310,000     |

After Target Encoding:
| City  |
|--------|
| 305,000 |
| 500,000 |
| 305,000 |

**Mathematics:**
\[ X_i = \frac{\sum_{y \in Y_{i}} y}{|Y_i|} \]

**Advantages:**
- Reduces dimensionality in high-cardinality data.

**Disadvantages:**
- Risk of data leakage; requires proper cross-validation.

---

### 5. **Frequency Encoding**
**Definition:**
Frequency Encoding assigns values based on the frequency of each category in the dataset.

**Use Cases:**
- Effective for handling categorical variables with many levels.

**Example:**
| City  |
|--------|
| A      |
| B      |
| A      |

After Frequency Encoding:
| City |
|-------|
| 2     |
| 1     |
| 2     |

**Mathematics:**
\[ X_i = \text{Frequency}(X_i) \]

**Advantages:**
- Efficient for high-cardinality data.
- Preserves some information about category importance.

**Disadvantages:**
- Does not capture relationships between categories and the target variable.

---

### 6. **Embedding Encoding**
**Definition:**
Embedding Encoding maps categorical variables into dense vector spaces, often using neural networks.

**Use Cases:**
- Highly effective for NLP tasks, deep learning models, and large categorical data with complex relationships.

**Example:**
A "Movie Genre" feature encoded into a 3D vector:
```
Action  -> [0.9, 0.1, 0.4]
Comedy  -> [0.2, 0.8, 0.3]
Drama   -> [0.1, 0.3, 0.9]
```

**Mathematics:**
- Learned embeddings are trained through backpropagation in neural networks.
- Each category \( X_i \) is mapped to a vector \( v_i \) in \( R^d \).

**Advantages:**
- Captures rich semantic information.
- Efficient for high-dimensional categorical data.

**Disadvantages:**
- Requires model training, making it computationally expensive.

