# Extreme Learning Machine in Golang
This is an implementation of the Extreme Learning Machine in Golang.

For explanation of ELM please refer [here](http://www.ntu.edu.sg/home/egbhuang/)

# Example
This example does three classifications of iris.
Thist is most simple example.

```go
package main

import (
  "fmt"

  elm "github.com/Yamashou/elm"
)

func main() {
  // Read iris datas
  X := elm.Iris()
  //Split training data and test data
  trainingDataSet, testDataSet := elm.TrainTestSplit(X, 0.4, 4, 3)
  e := elm.ELM{}
  //Trainig
  e.Fit(&trainingDataSet, 10)
  // Evaluate the model
  fmt.Println(e.Score(&testDataSet))
  // Evaluation results on feature quantity
  fmt.Println(e.GetResult([]float64{7.0, 3.2, 4.7, 1.4})) // -> 2
}
```
