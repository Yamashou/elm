package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func evaluationCheck(X mat.Dense, y []float64) {
	var x float64
	count := 0.0
	for i, v := range y {
		if X.At(i, 0) > 0 {
			x = 1
		} else {
			x = -1
		}
		if x == v {
			count++
			fmt.Println("true")
		} else {
			fmt.Println("false")
		}
	}
	fmt.Println(count / float64(len(y)))
}
