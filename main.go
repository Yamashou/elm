package main

import (
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type ELM struct {
	W     mat.Dense
	Beta  mat.Dense
	Train DataSet
	Test  DataSet
}

type DataSet struct {
	Data  [][]float64
	X     []float64
	Y     []float64
	XSize int
	YSize int
}

func main() {
	X := Iris()
	trainingDataSet, testDataSet := TrainTestSplit(X, 0.4, 4, 1)
	elm := ELM{}
	elm.Fit(&trainingDataSet, 10)
	elm.Score(&testDataSet)
}

func (d *DataSet) dataSplit() {
	k := 0
	for _, v := range d.Data {
		for _, vv := range v {
			if d.isData(k) {
				d.X = append(d.X, vv)
				k++
			} else {
				d.Y = append(d.Y, vv)
				k = 0
			}
		}
	}
}

func (d *DataSet) isData(k int) bool {
	return 0 <= k && k < d.XSize
}

func (d *DataSet) Set(data [][]float64, xSize, ySize int) {
	d.Data = data
	d.XSize = xSize
	d.YSize = ySize
}

func TrainTestSplit(d [][]float64, p float64, x, y int) (DataSet, DataSet) {
	var train [][]float64
	var test [][]float64
	for i := 0; i < len(d)-1; i++ {
		t := rand.Intn(len(d)-i) + i
		tmp := d[t]
		d[t] = d[i]
		d[i] = tmp
	}
	n := int(float64(len(d)) * (1 - p))
	for i, v := range d {
		if i < n {
			train = append(train, v)
		} else {
			test = append(test, v)
		}
	}
	var trainingDataSet DataSet
	var testDataSet DataSet
	trainingDataSet.Set(train, x, y)
	testDataSet.Set(test, x, y)

	trainingDataSet.dataSplit()
	testDataSet.dataSplit()
	return trainingDataSet, testDataSet
}

func (e *ELM) Fit(d *DataSet, hidNum int) {
	var data mat.Dense

	xArray := e.getAddBiasArray(d)
	rundArray := getRundomArray(hidNum, d.XSize+1)
	yArray := mat.NewDense(len(d.Y), d.YSize, d.Y)
	e.W = *rundArray

	H := e.getWeightMatrix(*xArray)
	data.Mul(H.T(), yArray)
	e.Beta = data
}

func (e *ELM) Score(d *DataSet) {
	var data mat.Dense

	testArray := e.getAddBiasArray(d)
	data.Mul(&e.W, testArray.T())

	gData := setSigmoid(data)
	var data2 mat.Dense
	data2.Mul(gData.T(), &e.Beta)
	evaluationCheck(data2, d.Y)
}

func (e *ELM) getAddBiasArray(d *DataSet) *mat.Dense {
	dataSize := d.XSize
	t := addBias(d.X, len(d.X)/dataSize, dataSize)
	return mat.NewDense(len(t)/(dataSize+1), dataSize+1, t)
}
