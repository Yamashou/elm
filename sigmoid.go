package elm

import (
	"math"

	"gonum.org/v1/gonum/mat"
)

func setSigmoid(x mat.Dense) *mat.Dense {
	var v float64
	n, m := x.Caps()
	k := 0
	result := make([]float64, n*m)
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			v = x.At(i, j)
			result[k] = sigmoid(v)
			k++
		}
	}
	return mat.NewDense(n, m, result)
}

func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}
