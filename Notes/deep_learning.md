Deep Learning Notes
---
A **Perceptron** is the simplest form of a neural network model, often considered the building block of more complex neural networks like multilayer perceptrons and deep learning models. It is a type of **binary classifier** that maps an input vector to an output using a simple mathematical model.

### Key Concepts of the Perceptron:

1. **Inputs**: The perceptron receives inputs (features) from a dataset.
2. **Weights**: Each input is assigned a weight that determines its importance.
3. **Weighted Sum**: The inputs are multiplied by their respective weights and summed.
4. **Activation Function**: The sum is passed through an activation function to produce an output.
5. **Bias**: An additional term is added to shift the output, allowing for more flexibility.

### Structure of a Perceptron:

- **Inputs**: ( x_1, x_2, ..., x_n ) (features of the data)
- **Weights**: ( w_1, w_2, ..., w_n ) (weights corresponding to the inputs)
- **Bias**: ( b ) (a constant term added to the weighted sum)
- **Output**: ( y ) (the final classification result)

The perceptron computes a **weighted sum** of the inputs, adds the bias, and applies an **activation function** to produce an output. The formula for the weighted sum is:

$$
z = \sum_{i=1}^{n} w_i x_i + b
$$

Where:
- ( w_i ) are the weights,
- ( x_i ) are the inputs,
- ( b ) is the bias.

### 1. **Weighted Sum** (Net Input)
The weighted sum (or linear combination) is calculated as:

$$
z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b
$$

This **z** value is a linear combination of the inputs and their corresponding weights. It essentially tells how strongly the inputs contribute to the final output.

### 2. **Activation Function**
The **activation function** determines the output of the perceptron based on the value of the weighted sum ( z ). For the perceptron, we typically use the **step function**, also known as the **Heaviside function**, for binary classification. It outputs either 0 or 1 depending on whether ( z ) is greater than or less than a certain threshold (usually 0).

The step function is:

$$
y = 
\begin{cases}
1 & \text{if } z \geq 0 \\
0 & \text{if } z < 0
\end{cases}
$$

This means:
- If the weighted sum ( z ) is positive or zero, the output will be **1** (positive class).
- If the weighted sum ( z ) is negative, the output will be **0** (negative class).

### Full Perceptron Formula:

$$
y = 
\begin{cases}
1 & \text{if } \sum_{i=1}^{n} w_i x_i + b \geq 0 \\
0 & \text{if } \sum_{i=1}^{n} w_i x_i + b < 0
\end{cases}
$$

In this equation:
- **Inputs** ( x_1, x_2, ..., x_n ) are features (e.g., pixel values of an image, measurements in a dataset).
- **Weights** ( w_1, w_2, ..., w_n ) are the coefficients that adjust the influence of each feature.
- **Bias** ( b ) is a constant term that shifts the decision boundary.

### 3. **Bias Term**
The **bias** ( b ) helps control the position of the decision boundary. Without the bias, the decision boundary is forced to pass through the origin, which might not always be the best fit for the data. The bias gives more flexibility by shifting the boundary to better separate the data.

### Perceptron Example:
Let's consider a simple binary classification example with two inputs.

#### Example:
Suppose you have two inputs, ( x_1 ) and ( x_2 ), with weights ( w_1 = 0.5 ) and ( w_2 = -0.6 ), and a bias of ( b = 0.2 ).

The formula becomes:
$$
z = 0.5 \cdot x_1 + (-0.6) \cdot x_2 + 0.2
$$

Now, let’s input the values ( x_1 = 1 ) and ( x_2 = 2 ):

$$
z = 0.5 \cdot 1 + (-0.6) \cdot 2 + 0.2 = 0.5 - 1.2 + 0.2 = -0.5
$$

Now, apply the step function:
$$
y = 
\begin{cases}
1 & \text{if } z \geq 0 \\
0 & \text{if } z < 0
\end{cases}
$$

Since ( z = -0.5 ), which is less than 0, the output is ( y = 0 ). Therefore, the perceptron classifies this input as belonging to the negative class (0).

### Training a Perceptron
During training, a perceptron uses a method called **gradient descent** (or in simpler cases, **perceptron learning algorithm**) to adjust its weights based on the error between the predicted and actual output. The update rule for weights is:

$$
w_i = w_i + \Delta w_i
$$
Where the weight update ( \Delta w_i ) is computed as:

$$
\Delta w_i = \eta \cdot (y_{\text{true}} - y_{\text{predicted}}) \cdot x_i
$$

- ( \eta ) is the **learning rate**, which controls how much the weights change in each step.
- ( y_{\text{true}} ) is the actual label of the data point.
- ( y_{\text{predicted}} ) is the output of the perceptron.

### Perceptron as a Linear Classifier
The perceptron can only solve linearly separable problems because its decision boundary is a **hyperplane**. It struggles with problems like XOR, where the classes are not linearly separable, which led to the development of more complex neural networks.

### Example of Linearly Separable Problem:
Consider the problem of classifying points in 2D space into two categories (positive or negative) based on their location. If the points can be separated by a straight line, a perceptron can learn the weights to classify them correctly.

### Limitations of Perceptron:
- **Linearly Separable Data**: A perceptron can only classify data that is linearly separable. If the classes cannot be separated by a straight line, the perceptron will fail.
- **No Non-linearity**: It uses a step function as the activation function, which is not differentiable. This limits its ability to learn more complex patterns.

### Summary:
- The **perceptron** is the simplest form of a neural network model and serves as a linear binary classifier.
- It works by calculating a **weighted sum** of its inputs, adding a **bias**, and passing the result through an **activation function** (step function).
- **Weights** and **bias** are adjusted during training based on errors in classification using a **learning rule**.

This simple yet powerful idea laid the foundation for modern neural networks and deep learning models.
---

A **Convolutional Neural Network (CNN)** is a type of deep learning model specifically designed for processing grid-like data, such as images. It has proven highly effective in image recognition, classification, and various other computer vision tasks. CNNs automatically learn spatial hierarchies of features from input images by applying filters in multiple layers.

### Key Concepts in CNNs:

1. **Convolution Operation**
2. **Filters/Kernels**
3. **Activation Function**
4. **Pooling/Downsampling**
5. **Fully Connected Layers**
6. **Training and Backpropagation**

### 1. **Convolution Operation**
The core idea behind CNNs is the **convolution** operation. In convolution, a small matrix called a **filter** or **kernel** slides across the input image to detect patterns, such as edges, textures, or colors.

#### Example:
Suppose you have a grayscale image of size 5x5 pixels:

```
Input Image (5x5):
[[10, 20, 30, 40, 50],
 [60, 70, 80, 90, 100],
 [110, 120, 130, 140, 150],
 [160, 170, 180, 190, 200],
 [210, 220, 230, 240, 250]]
```

Now, let’s say you have a 3x3 filter (kernel):

```
Filter (3x3):
[[1, 0, -1],
 [1, 0, -1],
 [1, 0, -1]]
```

The filter slides over the image and performs an element-wise multiplication with the section of the image it's currently covering. The result of the element-wise multiplication is summed up to produce a single value. This value is stored in a new output matrix called a **feature map** or **convolved feature**.

### 2. **Filters/Kernels**
A **filter** is a small matrix of weights that captures features like edges, textures, or other patterns in an image. Multiple filters are applied in a CNN to capture different features. In the early layers, filters detect basic patterns (e.g., edges), and in deeper layers, they capture more complex patterns (e.g., shapes, objects).

Filters are **learned** during training. The network adjusts the weights of these filters using backpropagation to optimize feature detection for a given task.

### 3. **Activation Function**
After convolution, an activation function (commonly **ReLU**, or Rectified Linear Unit) is applied to introduce non-linearity. Without non-linearity, the CNN would behave like a linear model, which cannot learn complex patterns.

- **ReLU** is defined as:
  $$
  \text{ReLU}(x) = \max(0, x)
  $$
This helps the network to focus on positive values and discard negative ones, which introduces the necessary non-linear properties to learn complex patterns.

#### Example of ReLU Activation:
```
Convolved feature (pre-activation):
[[5, -10],
 [-3, 7]]

After ReLU:
[[5, 0],
 [0, 7]]
```

### 4. **Pooling/Downsampling**
After convolution and activation, CNNs typically use a **pooling** layer to downsample the feature map. This reduces the spatial size, reducing the computational load and helping to extract dominant features.

- **Max Pooling** is the most common type, where the maximum value is taken from each region of the feature map.
- **Average Pooling** takes the average value instead.

#### Example of Max Pooling (2x2 filter with stride 2):
```
Input:
[[1, 3, 2, 4],
 [5, 6, 7, 8],
 [9, 2, 3, 1],
 [0, 1, 5, 2]]

Max Pooling result:
[[6, 8],
 [9, 5]]
```
Max pooling extracts the most important feature (highest value) in each 2x2 block.

### 5. **Fully Connected Layers**
Once convolution and pooling operations have been applied, the resulting feature map is flattened into a 1D vector. This vector is then passed to one or more **fully connected layers** (dense layers), where every node is connected to every other node. These layers act as a standard neural network that makes predictions.

#### Example:
If you have a 4x4 feature map after pooling:
```
Feature map:
[[1, 0, 2, 3],
 [4, 6, 5, 2],
 [0, 1, 3, 1],
 [5, 2, 2, 0]]

Flattened vector: [1, 0, 2, 3, 4, 6, 5, 2, 0, 1, 3, 1, 5, 2, 2, 0]
```

The fully connected layers use this flattened vector to classify the image.

### 6. **Training and Backpropagation**
CNNs are trained through **backpropagation**, where errors from the final prediction are sent backward through the network to update the weights of the filters and neurons. This process is done over several iterations (epochs) to minimize the error.

The loss function typically used for image classification is **categorical cross-entropy**.

### Example CNN for Image Classification (Simplified)
Let's assume we want to classify images of cats and dogs (binary classification).

1. **Input**: 64x64 color image (3 channels for RGB).
2. **Convolution Layer 1**: Apply a set of filters (e.g., 32 filters of size 3x3). The result is a set of feature maps, each highlighting specific features in the image.
3. **ReLU Activation**: Apply ReLU to introduce non-linearity.
4. **Max Pooling**: Downsample the feature maps by pooling (e.g., 2x2 filter).
5. **Convolution Layer 2**: Apply another set of filters (e.g., 64 filters of size 3x3).
6. **ReLU Activation**: Again, apply non-linearity.
7. **Max Pooling**: Pool the result to reduce dimensionality.
8. **Fully Connected Layer**: Flatten the pooled feature maps into a vector and pass them to a fully connected layer.
9. **Output**: Use a sigmoid function to output a probability for the two classes (cat or dog).

### Why CNNs Are Effective for Images:
- **Local Connectivity**: Filters are small and learn local patterns, such as edges or textures.
- **Parameter Sharing**: The same filter is applied across different regions of the image, reducing the number of parameters and ensuring that the network learns useful global features.
- **Translation Invariance**: Pooling and convolution operations make the model robust to small shifts and distortions in the image.

### Example Use Cases:
1. **Image Classification**: Recognizing whether an image contains a cat, dog, or other objects (e.g., using the popular ImageNet dataset).
2. **Object Detection**: Identifying and locating objects within an image (e.g., using YOLO or Faster R-CNN).
3. **Face Recognition**: Detecting and verifying faces in an image (e.g., using VGGFace).
4. **Medical Image Analysis**: Detecting tumors or abnormalities in medical scans (e.g., using CNNs in MRI or X-ray image analysis).

CNNs have revolutionized computer vision tasks due to their ability to learn and recognize complex patterns in images.