package elm

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

func evaluationCheck(X mat.Dense, y []float64) {
	n, m := X.Caps()
	x := make([][]float64, n*m)
	count := float64(n)
	for i := 0; i < n; i++ {
		data := make([]float64, m)
		for j := 0; j < m; j++ {
			if X.At(i, j) > 0 {
				data[j] = 1.0
			} else {
				data[j] = -1.0
			}
		}
		x[i] = data
	}
	k := 0
	flag := false
	for _, v := range x {
		flag = false
		for _, vv := range v {
			if vv != y[k] && !flag {
				flag = true
				count--
			}
			k++
		}
	}
	fmt.Println(count / float64(len(y)/m))
}
