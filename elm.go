package elm

import (
	"math/rand"

	"github.com/Yamashou/mpinverse"
	"gonum.org/v1/gonum/mat"
)

type ELM struct {
	W    mat.Dense
	Beta mat.Dense
}

func (e *ELM) getWeightMatrix(X mat.Dense) mat.Dense {
	var b mat.Dense
	b.Mul(&e.W, X.T())
	return mpinverse.NewMPInverse(SetSigmoid(b))
}

func (e *ELM) Fit(d *DataSet, hidNum int) {
	var data mat.Dense

	xArray := e.GetAddBiasArray(d)
	rundArray := getRundomArray(hidNum, d.XSize+1)
	yArray := mat.NewDense(len(d.Y)/d.YSize, d.YSize, d.Y)
	e.W = *rundArray

	H := e.getWeightMatrix(*xArray)
	data.Mul(H.T(), yArray)
	e.Beta = data
}

func (e *ELM) Score(d *DataSet) float64 {
	var data mat.Dense

	testArray := e.GetAddBiasArray(d)
	data.Mul(&e.W, testArray.T())

	gData := SetSigmoid(data)
	var result mat.Dense
	result.Mul(gData.T(), &e.Beta)
	return evaluationCheck(result, d.Y)
}

func (e *ELM) GetAddBiasArray(d *DataSet) *mat.Dense {
	dataSize := d.XSize
	t := addBias(d.X, len(d.X)/dataSize, dataSize)
	return mat.NewDense(len(t)/(dataSize+1), dataSize+1, t)
}

func (e *ELM) GetResult(d []float64) int {
	d = append(d, 1)
	vec := mat.NewDense(1, len(d), d)
	var hneuron mat.Dense
	hneuron.Mul(&e.W, vec.T())
	gData := SetSigmoid(hneuron)
	var X mat.Dense
	X.Mul(gData.T(), &e.Beta)

	n, m := X.Caps()
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			if X.At(i, j) > 0 {
				return j
			}
		}
	}
	return -1
}

func getRundomArray(n, m int) *mat.Dense {
	data := make([]float64, n*m)
	for i := range data {
		data[i] = rand.NormFloat64() / 10
	}
	return mat.NewDense(n, m, data)
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
