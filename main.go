package main

import (
	"fmt"
	"math"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type ELM struct {
	W    mat.Dense
	Data mat.Dense
}

func main() {
	X := Iris()
	var train [][]float64
	var test [][]float64
	for i, v := range X {
		if i%3 == 2 {
			test = append(test, v)
		} else {
			train = append(train, v)
		}
	}

	var trainX []float64
	var trainY []float64
	var testX []float64
	var testY []float64
	k := 0
	for _, v := range train {
		for _, vv := range v {
			if k != 3 {
				trainX = append(trainX, vv)
				k++
			} else {
				trainY = append(trainY, vv)
				k = 0
			}
		}
	}
	k = 0
	for _, v := range test {
		for _, vv := range v {
			if k != 3 {
				testX = append(testX, vv)
				k++
			} else {
				testY = append(testY, vv)
				k = 0
			}
		}
	}
	var data mat.Dense
	t := addBias(trainX, len(trainX)/3, 3)
	xArray := mat.NewDense(len(t)/4, 4, t)
	yArray := mat.NewDense(len(trainY), 1, trainY)
	rundArray := getRundomArray(10, 4)
	elm := ELM{}
	elm.W = *rundArray
	H := elm.getWeightMatrix(*xArray)
	data.Mul(H.T(), yArray)
	elm.Data = data
	fm := mat.Formatted(&data, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("m = %4.2f", fm)
	var data2 mat.Dense
	var data4 mat.Dense
	t2 := addBias(testX, len(testX)/3, 3)
	xArray2 := mat.NewDense(len(t2)/4, 4, t2)
	data2.Mul(xArray2, elm.W.T())
	fmt.Println(data2.Caps())
	H2 := elm.getWeightMatrix(*xArray2)
	data4.Mul(&H2, &data)
	fmm := mat.Formatted(&data4, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("m = %4.2f", fmm)
}

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
func getMPInverse(a *mat.Dense) mat.Dense {
	var svd mat.SVD
	svd.Factorize(a, mat.SVDThin)
	svdV := svd.VTo(nil)
	svdU := svd.UTo(nil)
	svdS := svd.Values(nil)

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
	var ib mat.Dense
	ib.Mul(svdV, b)
	return ib
}

func setSigmoid(x mat.Dense) *mat.Dense {
	var v float64
	n, m := x.Caps()
	k := 0
	result := make([]float64, n*m)
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			v = x.At(i, j)
			result[k] = 1.0 / (1.0 + math.Exp(-v))
			k++
		}
	}
	return mat.NewDense(n, m, result)
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

func addBias(X []float64, n, m int) []float64 {
	result := make([]float64, n*(m+1))
	k := 0
	count := 0
	for _, v := range X {
		result[k] = v
		if count == (m - 1) {
			result[k+1] = 1.0
			k += 2
			count = 0
			continue
		}
		k++
		count++
	}
	return result
}
