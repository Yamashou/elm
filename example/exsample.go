package main

import (
	"fmt"

	elm "github.com/Yamashou/elm"
)

func main() {
	X := elm.Iris()
	trainingDataSet, testDataSet := elm.TrainTestSplit(X, 0.4, 4, 3)
	e := elm.ELM{}
	e.Fit(&trainingDataSet, 10)
	fmt.Println(e.Score(&testDataSet))
	elm.CrossValidation(X, 3, 4, 3, 10)
}
