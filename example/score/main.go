package main

import (
	"fmt"

	elm "github.com/Yamashou/elm"
)

func main() {
	X := elm.Iris()
	trainingDataSet, testDataSet := elm.TrainTestSplit(X, 0.4, 4, 3)
	e := elm.ELM{}
	e.Fit(&trainingDataSet, 10, 0)
	fmt.Println(e.Score(&testDataSet))
	fmt.Println(e.GetResult([]float64{7.0, 3.2, 4.7, 1.4})) // -> 2

}
