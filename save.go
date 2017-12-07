package elm

import (
	"os"
)

func (e ELM) MarshalBinaryTo(name string) (int, error) {
	v1, _ := os.Create("iris_w_" + name)
	defer v1.Close()
	v2, _ := os.Create("iris_beta_" + name)
	defer v2.Close()

	n1, err := e.W.MarshalBinaryTo(v1)
	if err != nil {
		return n1, err
	}
	n2, err := e.Beta.MarshalBinaryTo(v2)
	if err != nil {
		return n1 + n2, err
	}
	return n1 + n2, nil
}
