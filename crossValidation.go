package elm

import "fmt"

func CrossValidation(d [][]float64, n, x, y, m int) {
	data := make([][][]float64, n)
	d = mixData(d)
	for i, v := range d {
		data[i%n] = append(data[i%n], v)
	}
	count := 0.0
	for i := 0; i < n; i++ {
		var train [][]float64
		var test [][]float64
		e := ELM{}
		for j := 0; j < n; j++ {
			if j == i {
				continue
			}
			train = append(train, data[j]...)
		}
		test = data[i]
		var trainingDataSet DataSet
		var testDataSet DataSet
		trainingDataSet.Set(train, x, y)
		testDataSet.Set(test, x, y)

		trainingDataSet.dataSplit()
		testDataSet.dataSplit()
		e.Fit(&trainingDataSet, m)
		count += e.Score(&testDataSet)
	}
	fmt.Println(count / float64(n))
}
