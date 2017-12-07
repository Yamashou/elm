package main

import (
	"github.com/Yamashou/elm"
)

func main() {
	X := elm.Iris()
	trainingDataSet, _ := elm.TrainTestSplit(X, 0.4, 4, 3)
	e := elm.ELM{}
	e.Fit(&trainingDataSet, 10)
	e.MarshalBinaryTo("test")
}
