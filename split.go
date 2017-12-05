package elm

func TrainTestSplit(d [][]float64, p float64, x, y int) (DataSet, DataSet) {
	var train [][]float64
	var test [][]float64
	d = mixData(d)
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
