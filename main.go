package main

import (
	"fmt"

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
			if k != 4 {
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
			if k != 4 {
				testX = append(testX, vv)
				k++
			} else {
				testY = append(testY, vv)
				k = 0
			}
		}
	}
	var data mat.Dense
	t := addBias(trainX, len(trainX)/4, 4)
	xArray := mat.NewDense(len(t)/5, 5, t)
	yArray := mat.NewDense(len(trainY), 1, trainY)
	rundArray := getRundomArray(10, 5)
	elm := ELM{}
	elm.W = *rundArray
	H := elm.getWeightMatrix(*xArray)
	data.Mul(H.T(), yArray)
	elm.Data = data
	fm := mat.Formatted(&data, mat.Prefix("    "), mat.Squeeze())
	fmt.Printf("β = %4.2f\n", fm)
	var data2 mat.Dense
	tt := addBias(testX, len(testX)/4, 4)
	testArray := mat.NewDense(len(tt)/5, 5, tt)
	data2.Mul(rundArray, testArray.T())
	gData := setSigmoid(data2)
	var data3 mat.Dense
	data3.Mul(gData.T(), &data)
	evaluationCheck(data3, testY)
}
