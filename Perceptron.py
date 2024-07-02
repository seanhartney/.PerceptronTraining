# Import Dataset below


# Works as an activator function for the Perceptron
def unit_step(x):
  if x < 0:
    return 0
  return 1

# This function assumes takes two equal length lists assuming one contains
# predicted values and the other contains true values
def accuracy_score(y1, y2):
  if len(y1) != len(y2):
    raise Exception("Dimensions don't match")
  num_matches = 0
  for i in range(len(y1)):
    if y1[i] == y2[i]:
      num_matches += 1
  return num_matches / len(y1)

class Perceptron:
  def __init__(self, num_features):
    self.weights = [0] * num_features
    self.bias = 0

# Returns a 1 or 0 which is a prediction that the perceptron makes for a single data point
  def predict_sample(self, x):
    total = self.bias
    for i in range(len(x)):
      total += self.weights[i] * x[i]
    return unit_step(total)

# Makes predictions similar to predict_sample but for a whole list
  def predict(self, X):
    return [self.predict_sample(x) for x in X]

# The function that trains on the data we provide
# The learning rate is mutable as well as the iterations, you can change the defaults or set them manually
  def train(self, X_train, y_train, learning_rate=.001, iterations=200):
    for iter in range(iterations):
      y_hat = self.predict(X_train)
      accuracy = accuracy_score(y_hat, y_train)
      #print("Accuracy after", iter, "epochs:", accuracy) This line is optional to see each stage accuracy
      for j in range(len(X_train)):
        if y_hat[j] != y_train[j]:
          for i in range(len(self.weights)):
            self.weights[i] += learning_rate * (y_train[j] - y_hat[j]) * X_train[j][i]
          self.bias += learning_rate * (y_train[j] - y_hat[j])


# Code Testing to make sure everything runs correctly 

# Accuracy Score Function
assert(accuracy_score([0,1,0,1,0,1], [1,0,1,0,1,0]) == 0)
assert(accuracy_score([1,0,1,0], [0,1,1,0]) == .5)
print("Accuracy Score Tests Passed")

# Class Setup
clf_test1 = Perceptron(2)
assert(clf_test1.weights == [0,0] and clf_test1.bias == 0)
clf_test2 = Perceptron(5)
assert(clf_test2.weights == [0,0,0,0,0] and clf_test2.bias == 0)
print("Class Setup Tests Passed")

# Prediction Samples
clf_test1 = Perceptron(2)
clf_test1.weights = [-2,1]
clf_test1.bias = 3
assert(clf_test1.predict_sample([0, 0]) == 1)
assert(clf_test1.predict_sample([2, 0]) == 0)
assert(clf_test1.predict_sample([0, 3]) == 1)
assert(clf_test1.predict_sample([3, 2]) == 0)
clf_test2 = Perceptron(5)
clf_test2.weights = [-2,1,4,-1,3]
clf_test2.bias = -2
assert(clf_test2.predict_sample([0,0,0,0,0]) == 0)
assert(clf_test2.predict_sample([1,0,2,0,3]) == 1)
assert(clf_test2.predict_sample([0,4,0,3,0]) == 0)
assert(clf_test2.predict_sample([5,2,0,6,0]) == 0)
print("The Prediction Sample Tests Passed")

# Predict Function test
clf_test1 = Perceptron(2)
clf_test1.weights = [-2,1]
clf_test1.bias = 3
assert(clf_test1.predict([[0, 0], [2, 0], [0, 3]]) == [1, 0, 1])
clf_test2 = Perceptron(5)
clf_test2.weights = [-2,1,4,-1,3]
clf_test2.bias = -2
assert(clf_test2.predict([[0,0,0,0,0], [1,0,2,0,3], [0,4,0,3,0], [5,2,0,6,0]]) == [0, 1, 0, 0])
print("Predict Function Tests Passed")

# Training Function Test
clf_test1 = Perceptron(2)
clf_test1.train([[0, 0], [2, 0], [0, 3]], [0, 0, 0], learning_rate = 0.1, iterations = 1)
assert([round(w, 4) for w in clf_test1.weights] == [-0.2, -0.3] and round(clf_test1.bias, 4) == -0.3)
clf_test2 = Perceptron(2)
clf_test2.train([[0, 0], [2, 0], [0, 3]], [1, 0, 0], learning_rate = 0.01, iterations = 5)
assert(clf_test2.weights == [-0.02, -0.03] and clf_test2.bias == 0)
print("Training Tests Passed")