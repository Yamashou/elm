package main

import "gonum.org/v1/gonum/mat"

func listToArray(svdS []float64, utn, utm int) *mat.Dense {
	S := make([]float64, utn*utm)
	k := 0
	for i := 0; i < utn; i++ {
		for j := 0; j < utm; j++ {
			S[k] = svdS[i]
			k++
		}
	}
	return mat.NewDense(utn, utm, S)
}
