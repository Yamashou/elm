package elm

import "math/rand"

func mixData(d [][]float64) [][]float64 {
	for i := 0; i < len(d)-1; i++ {
		t := rand.Intn(len(d)-i) + i
		tmp := d[t]
		d[t] = d[i]
		d[i] = tmp
	}
	return d
}
