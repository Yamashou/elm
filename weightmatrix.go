package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

func (e ELM) getWeightMatrix(X mat.Dense) mat.Dense {
	var b mat.Dense
	b.Mul(&e.W, X.T())
	return getMPInverse(setSigmoid(b))
}
func getRundomArray(n, m int) *mat.Dense {
	data := make([]float64, n*m)
	for i := range data {
		data[i] = rand.NormFloat64() / 10
	}
	return mat.NewDense(n, m, data)
}
