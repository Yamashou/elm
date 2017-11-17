package main

import (
	"encoding/csv"
	"io"
	"log"
	"os"
	"strconv"

	"golang.org/x/text/encoding/japanese"
	"golang.org/x/text/transform"
)

func failOnError(err error) {
	if err != nil {
		log.Fatal("Error:", err)
	}
}

func Iris() [][]float64 {
	var x []float64
	var y [][]float64
	file1, err := os.Open("/Users/yamamshou/go/src/github.com/sjwhitworth/golearn/examples/datasets/iris.csv")
	failOnError(err)
	defer file1.Close()

	reader := csv.NewReader(transform.NewReader(file1, japanese.ShiftJIS.NewDecoder()))
	reader.LazyQuotes = true // ダブルクオートを厳密にチェックしない

	log.Printf("Start")
	for {
		record, err := reader.Read() // 1行読み出す
		if err == io.EOF {
			break
		} else {
			failOnError(err)
		}
		t := map[string]float64{"Iris-setosa": 1.0, "Iris-versicolor": -1.0, "Iris-virginica": 0.0}
		for i, v := range record {

			if i > 0 {
				if i != 4 {
					f64, _ := strconv.ParseFloat(v, 64)
					x = append(x, f64)
				} else {
					x = append(x, t[v])
				}
			}
			y = append(y, x)
		}
	}
	log.Printf("Finish !")
	return y
}
