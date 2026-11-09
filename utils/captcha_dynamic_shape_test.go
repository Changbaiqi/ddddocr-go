package utils

import (
	"encoding/json"
	"fmt"
	"image"
	"image/color"
	"image/draw"
	"os"
	"strings"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

func TestCaptchaModelsAllocateActualOutputShape(t *testing.T) {
	// Go 1.23 does not have testing.T.Chdir. Do not run this test in parallel.
	oldDir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(t.TempDir()); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.Chdir(oldDir); err != nil {
			t.Error(err)
		}
	})
	DDDDOcrCoreInit()
	t.Cleanup(func() {
		if err := ort.DestroyEnvironment(); err != nil {
			t.Error(err)
		}
	})

	var charset []string
	if err := json.Unmarshal([]byte(getCharCode()), &charset); err != nil {
		t.Fatal(err)
	}
	for _, width := range []int{128, 192, 193, 199, 200, 201, 256} {
		t.Run(fmt.Sprint(width), func(t *testing.T) {
			img := image.NewGray(image.Rect(0, 0, width, 64))
			draw.Draw(img, img.Bounds(), image.NewUniform(color.White), image.Point{}, draw.Src)
			for i := 0; i < 4; i++ {
				x := 8 + i*(width-16)/4
				draw.Draw(img, image.Rect(x, 16, x+8, 49), image.NewUniform(color.Black), image.Point{}, draw.Src)
			}
			input, err := ort.NewTensor[float32](ort.NewShape(1, 1, 64, int64(width)), ImageToGrayFloatArray(img))
			if err != nil {
				t.Fatal(err)
			}
			defer input.Destroy()
			beta, err := runCaptchaModel[float32](betaModelPath, "387", input)
			if err != nil {
				t.Fatal(err)
			}
			defer beta.Destroy()
			shape := beta.GetShape()
			if len(shape) != 3 || shape[0] <= 0 || shape[1] != 1 || shape[2] != betaClasses || int64(len(beta.GetData())) != shape[0]*betaClasses {
				t.Fatalf("beta shape: %v", shape)
			}
			if _, err := BetaOCRVerification(img); err != nil {
				t.Fatal(err)
			}

			legacy, err := runCaptchaModel[int64]("./assets/common_old.onnx", "output", input)
			if err != nil {
				t.Fatal(err)
			}
			defer legacy.Destroy()
			shape = legacy.GetShape()
			if len(shape) != 2 || shape[0] != 1 || shape[1] <= 0 || int64(len(legacy.GetData())) != shape[1] {
				t.Fatalf("legacy shape: %v", shape)
			}
			var expected strings.Builder
			for _, id := range legacy.GetData() {
				if id < 0 || id >= int64(len(charset)) {
					t.Fatalf("invalid character id: %d", id)
				}
				if id != 0 {
					expected.WriteString(charset[id])
				}
			}
			if got := AutoOCRVerification(img); got != expected.String() {
				t.Fatalf("legacy decode = %q, want %q", got, expected.String())
			}
			t.Logf("beta %v, legacy %v", beta.GetShape(), legacy.GetShape())
		})
	}
	t.Run("wrong_output_type", func(t *testing.T) {
		input, err := ort.NewTensor[float32](ort.NewShape(1, 1, 64, 128), make([]float32, 64*128))
		if err != nil {
			t.Fatal(err)
		}
		defer input.Destroy()
		if output, err := runCaptchaModel[int64](betaModelPath, "387", input); err == nil || output != nil {
			t.Fatal("wrong output type accepted")
		}
	})
	for _, img := range []image.Image{nil, image.NewGray(image.Rect(0, 0, 0, 64)), image.NewGray(image.Rect(0, 0, 64, 0))} {
		if _, err := BetaOCRVerification(img); err == nil {
			t.Error("invalid image accepted by beta")
		}
		if got := AutoOCRVerification(img); got != "" {
			t.Error("invalid image accepted by legacy")
		}
	}
}
