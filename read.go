package elm

import (
	"io"
	"os"

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

func UnmarshalBinaryFromName(name string) (ELM, error) {
	w, err := os.Open("./w_" + name)
	if err != nil {
		return ELM{}, err
	}
	defer w.Close()

	b, err := os.Open("./beta_" + name)
	if err != nil {
		return ELM{}, err
	}
	defer b.Close()

	return UnmarshalBinaryFrom(w, b)
}

func UnmarshalBinary(w, b []byte) (ELM, error) {
	var W, Beta mat.Dense

	err := W.UnmarshalBinary(w)
	if err != nil {
		return ELM{}, err
	}

	err = Beta.UnmarshalBinary(b)
	if err != nil {
		return ELM{}, err
	}

	return ELM{W, Beta}, nil
}
