package main

import elm "github.com/Yamashou/elm"

func main() {
	X := elm.Iris()
	trainingDataSet, testDataSet := elm.TrainTestSplit(X, 0.4, 4, 1)
	e := elm.ELM{}
	e.Fit(&trainingDataSet, 10)
	e.Score(&testDataSet)
}
