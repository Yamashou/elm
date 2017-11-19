package main

import "math"

func getCutoff(svdS []float64) float64 {
	v1 := svdS[0]
	for _, v := range svdS {
		v1 = math.Max(v, v1)
	}
	return 1e-15 * v1
}
