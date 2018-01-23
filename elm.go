package elm

import (
	"math/rand"
	"time"

	"github.com/Yamashou/mpinverse"
	"gonum.org/v1/gonum/mat"
)

// ELM is a learning model type
type ELM struct {
	W    mat.Dense
	Beta mat.Dense
}

func (e *ELM) getWeightMatrix(X mat.Dense) mat.Dense {
	var b mat.Dense
	b.Mul(&e.W, X.T())
	return mpinverse.NewMPInverse(SetSigmoid(b))
}

// Fit is a learning function, d: Learning data, hidNum: Hidden neurons
func (e *ELM) Fit(d *DataSet, hidNum int, seed int64) int64 {
	var data mat.Dense

	xArray := e.GetAddBiasArray(d)
	if seed == 0 {
		seed = time.Now().UnixNano()
	}
	rundArray := getRundomArray(hidNum, d.XSize+1, seed)
	yArray := mat.NewDense(len(d.Y)/d.YSize, d.YSize, d.Y)
	e.W = *rundArray

	H := e.getWeightMatrix(*xArray)
	data.Mul(H.T(), yArray)
	e.Beta = data
	return seed
}

//Score returns the accuracy of the model, d: Test data
func (e *ELM) Score(d *DataSet) float64 {
	var data mat.Dense

	testArray := e.GetAddBiasArray(d)
	data.Mul(&e.W, testArray.T())

	gData := SetSigmoid(data)
	var result mat.Dense
	result.Mul(gData.T(), &e.Beta)
	return evaluationCheck(result, d.Y)
}

//GetAddBiasArray adds bias to data, d: data(learning, test, etc...)
func (e *ELM) GetAddBiasArray(d *DataSet) *mat.Dense {
	dataSize := d.XSize
	t := addBias(d.X, len(d.X)/dataSize, dataSize)
	return mat.NewDense(len(t)/(dataSize+1), dataSize+1, t)
}

// GetResult returns the evaluation result on certain data, d: feature vector
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

func getRundomArray(n, m int, seed int64) *mat.Dense {
	data := make([]float64, n*m)
	floadRand := rand.New(rand.NewSource(seed))
	for i := range data {
		data[i] = floadRand.NormFloat64() / 10
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
