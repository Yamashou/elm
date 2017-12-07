package elm

import (
	"io"

	"gonum.org/v1/gonum/mat"
)

func UnmarshalBinaryFrom(w, b io.Reader) (ELM, error) {
	var W, Beta mat.Dense

	_, err := W.UnmarshalBinaryFrom(w)
	if err != nil {
		return ELM{}, err
	}

	_, err = Beta.UnmarshalBinaryFrom(b)
	if err != nil {
		return ELM{}, err
	}

	return ELM{W, Beta}, nil
}
