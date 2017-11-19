package main

func main() {
	X := Iris()
	trainingDataSet, testDataSet := TrainTestSplit(X, 0.4, 4, 1)
	elm := ELM{}
	elm.Fit(&trainingDataSet, 10)
	elm.Score(&testDataSet)
}
