package main

import (
	"fmt"

	"gonum.org/v1/gonum/mat"
)

type ELM struct {
	W    mat.Dense
	Data mat.Dense
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

	var data mat.Dense
	t := addBias(trainingDataSet.X, len(trainingDataSet.X)/4, 4)
	xArray := mat.NewDense(len(t)/5, 5, t)
	yArray := mat.NewDense(len(trainingDataSet.Y), 1, trainingDataSet.Y)
	rundArray := getRundomArray(10, 5)
	elm := ELM{}
	elm.W = *rundArray
	H := elm.getWeightMatrix(*xArray)
	data.Mul(H.T(), yArray)
	elm.Data = data
	fm := mat.Formatted(&data, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("β = %4.2f\n", fm)
	var data2 mat.Dense
	tt := addBias(testDataSet.X, len(testDataSet.X)/4, 4)
	testArray := mat.NewDense(len(tt)/5, 5, tt)
	data2.Mul(rundArray, testArray.T())
	gData := setSigmoid(data2)
	var data3 mat.Dense
	data3.Mul(gData.T(), &data)
	evaluationCheck(data3, testDataSet.Y)
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
