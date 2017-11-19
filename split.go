package main

import "math/rand"

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
