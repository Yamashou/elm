package main

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

type Data struct {
	Num     float64
	Hid_num float64
}

func (data Data) sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-data.Num*x))
}
func main() {
	a := mat.NewDense(3, 2, []float64{
		1, 1,
		1, 1,
		1, 0,
	})
	b := getMPInverse(a)
	fc := mat.Formatted(&b, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("\nc = %v", fc)
}

func getMPInverse(a *mat.Dense) mat.Dense {
	var svd mat.SVD
	svd.Factorize(a, mat.SVDThin)
	svdV := svd.VTo(nil)
	svdU := svd.UTo(nil)
	svdS := svd.Values(nil)
	fmt.Println(svdS)
	cutoff := getCutoff(svdS)
	for i := range svdS {
		if svdS[i] > cutoff {
			svdS[i] = 1.0 / svdS[i]
		} else {
			svdS[i] = 0.0
		}
	}
	svdUt := svdU.T()
	utn, utm := svdUt.Dims()
	b := getSingularArray(svdS, utn, utm)
	b.MulElem(b, svdUt)
	fc := mat.Formatted(svdV, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("c = %v", fc)
	var ib mat.Dense
	ib.Mul(svdV, b)
	return ib

}

func getCutoff(svdS []float64) float64 {
	v1 := svdS[0]
	for _, v := range svdS {
		v1 = math.Max(v, v1)
	}
	return 1e-15 * v1
}

func getSingularArray(svdS []float64, utn, utm int) *mat.Dense {
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
