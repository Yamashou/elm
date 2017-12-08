package elm

import (
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/plotutil"
	"gonum.org/v1/plot/vg"
)

func Points(n []int, v []float64) plotter.XYs {
	pts := make(plotter.XYs, len(n))

	for i := range pts {
		pts[i].Y = v[i]
		pts[i].X = float64(n[i])
	}
	return pts
}

func PlotPng(X [][]float64, name string) {
	h := []int{5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100}
	ans := make([]float64, len(h))
	for i, v := range h {
		ans[i] = CrossValidation(X, 3, 256, 1, v)
	}

	p, err := plot.New()
	if err != nil {
		panic(err)
	}
	p.Title.Text = "Face"
	p.X.Label.Text = "Hidden neurons"
	p.Y.Label.Text = "percent"

	err = plotutil.AddLinePoints(p, "First", Points(h, ans))
	if err != nil {
		panic(err)
	}
	if err := p.Save(4*vg.Inch, 4*vg.Inch, name+".png"); err != nil {
		panic(err)
	}
}
