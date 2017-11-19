package main

import (
	"fmt"

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
	var train [][]float64
	var test [][]float64
	for i, v := range X {
		if i%3 == 2 {
			test = append(test, v)
		} else {
			train = append(train, v)
		}
	}
	var trainingDataSet DataSet
	var testDataSet DataSet
	trainingDataSet.Data = train
	trainingDataSet.XSize = 4
	trainingDataSet.YSize = 1

	testDataSet.Data = test
	testDataSet.XSize = 4
	testDataSet.YSize = 1

	trainingDataSet.dataSplit()
	testDataSet.dataSplit()
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

func (e *ELM) Fit(d *DataSet, hidNum int) {
	var data mat.Dense
	dataSize := d.XSize

	t := addBias(d.X, len(d.X)/dataSize, dataSize)
	fmt.Println(dataSize)
	xArray := mat.NewDense(len(t)/(dataSize+1), dataSize+1, t)
	rundArray := getRundomArray(hidNum, dataSize+1)
	yArray := mat.NewDense(len(d.Y), d.YSize, d.Y)

	e.W = *rundArray
	H := e.getWeightMatrix(*xArray)
	data.Mul(H.T(), yArray)
	e.Beta = data
}

func (e *ELM) Score(d *DataSet) {
	var data2 mat.Dense
	dataSize := d.XSize
	tt := addBias(d.X, len(d.X)/dataSize, dataSize)
	testArray := mat.NewDense(len(tt)/(dataSize+1), dataSize+1, tt)
	data2.Mul(&e.W, testArray.T())

	gData := setSigmoid(data2)
	var data3 mat.Dense
	data3.Mul(gData.T(), &e.Beta)
	evaluationCheck(data3, d.Y)
}
