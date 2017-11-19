package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func getRundomArray(n, m int) *mat.Dense {
	data := make([]float64, n*m)
	for i := range data {
		data[i] = rand.NormFloat64() / 10
	}
	return mat.NewDense(n, m, data)
}
