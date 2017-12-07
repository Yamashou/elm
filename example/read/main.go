package main

import (
	"fmt"
	"os"

	"github.com/Yamashou/elm"
)

func main() {
	w, err := os.Open("../save/iris_w_test")
	if err != nil {
		panic(err)
	}
	b, err := os.Open("../save/iris_beta_test")
	if err != nil {
		panic(err)
	}
	e, err := elm.UnmarshalBinaryFrom(w, b)
	X := elm.Iris()
	_, testDataSet := elm.TrainTestSplit(X, 0.4, 4, 3)
	fmt.Println(e.Score(&testDataSet))
}
