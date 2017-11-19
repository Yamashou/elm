package main

type DataSet struct {
	Data  [][]float64
	X     []float64
	Y     []float64
	XSize int
	YSize int
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
