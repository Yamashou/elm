package elm

import (
	"image"
	"io/ioutil"
	"net/http"
	"path/filepath"

	"github.com/PuerkitoBio/goquery"
	"github.com/disintegration/imaging"
	"github.com/kelvins/lbph"
	"github.com/kelvins/lbph/lbp"
)

func GetCharacteristic(p [][]uint64) []float64 {
	x := make([]float64, 256)
	for i := 0; i < len(x); i++ {
		x[i] = 0
	}
	for _, v := range p {
		for _, vv := range v {
			x[int(vv)]++
		}
	}
	return x
}

func GetLBH(m image.Image) [][]uint64 {
	params := lbph.Params{
		Radius:    1,
		Neighbors: 8,
		GridX:     8,
		GridY:     8,
	}

	pixels, err := lbp.Calculate(m, params.Radius, params.Neighbors)
	if err != nil {
		panic(err)
	}
	return pixels
}

func GetLocalImgPathToFeaturVector(paths []string, X [][]float64, ans []float64, max int) [][]float64 {
	for i, v := range paths {
		file, err := imaging.Open(v)
		if err != nil {
			continue
		}
		vv := GetCharacteristic(GetLBH(file))
		vv = append(vv, ans...)
		X = append(X, vv)
		if i == max {
			break
		}
	}
	return X
}

func GetPage(url string) []string {
	var images []string
	doc, _ := goquery.NewDocument(url)
	doc.Find("img").Each(func(_ int, s *goquery.Selection) {
		url, _ := s.Attr("src")
		images = append(images, url)
	})
	return images
}

func GetImgPathToFeaturVector(X [][]float64, urls []string, ans []float64) [][]float64 {
	for _, v := range urls {
		response, err := http.Get(v)
		if err != nil {
			panic(err)
		}
		defer response.Body.Close()
		img, _, err := image.Decode(response.Body)
		if err != nil {
			panic(err)
		}
		vv := GetCharacteristic(GetLBH(img))
		vv = append(vv, ans...)
		X = append(X, vv)
	}
	return X
}

func Dirwalk(dir string) []string {
	files, err := ioutil.ReadDir(dir)
	if err != nil {
		panic(err)
	}

	var paths []string
	for _, file := range files {
		if file.IsDir() {
			paths = append(paths, Dirwalk(filepath.Join(dir, file.Name()))...)
			continue
		}
		paths = append(paths, filepath.Join(dir, file.Name()))
	}

	return paths
}
