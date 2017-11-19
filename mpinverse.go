package main

import "gonum.org/v1/gonum/mat"

func getMPInverse(a *mat.Dense) mat.Dense {
	var svd mat.SVD
	svd.Factorize(a, mat.SVDThin)
	svdV := svd.VTo(nil)
	svdU := svd.UTo(nil)
	svdS := svd.Values(nil)

	cutoff := getCutoff(svdS)
	for i := range svdS {
		if svdS[i] > cutoff {
			svdS[i] = 1.0 / svdS[i]
		} else {
			svdS[i] = 0.0
		}
	}
	svdUt := svdU.T()
	utn, utm := svdUt.Dims()
	b := listToArray(svdS, utn, utm)
	b.MulElem(b, svdUt)
	var ib mat.Dense
	ib.Mul(svdV, b)
	return ib
}
