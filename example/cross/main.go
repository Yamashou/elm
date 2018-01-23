package main

import (
	"fmt"

	elm "github.com/Yamashou/elm"
)

func main() {
	X := elm.Iris()
	fmt.Println(elm.CrossValidation(X, 3, 4, 3, 10, 0))
}
