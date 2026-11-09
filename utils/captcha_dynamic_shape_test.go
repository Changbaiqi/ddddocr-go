package utils

import (
	"image"
	"image/color"
	"image/draw"
	"os"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

func TestCaptchaModelsAllocateActualOutputShape(t *testing.T) {
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
	defer ort.DestroyEnvironment()

	for _, width := range []int{128, 192, 200, 256} {
		img := image.NewGray(image.Rect(0, 0, width, 64))
		draw.Draw(img, img.Bounds(), image.NewUniform(color.White), image.Point{}, draw.Src)
		for i := 0; i < 4; i++ {
			x := 8 + i*(width-16)/4
			draw.Draw(img, image.Rect(x, 16, x+8, 49), image.NewUniform(color.Black), image.Point{}, draw.Src)
		}
		if _, err := BetaOCRVerification(img); err != nil {
			t.Fatalf("beta model at width %d: %v", width, err)
		}

		input, err := ort.NewTensor[float32](ort.NewShape(1, 1, 64, int64(width)), ImageToGrayFloatArray(img))
		if err != nil {
			t.Fatal(err)
		}
		output, err := runCaptchaModel[int64]("./assets/common_old.onnx", "output", input)
		input.Destroy()
		if err != nil {
			t.Fatalf("legacy model at width %d: %v", width, err)
		}
		outputLen := len(output.GetData())
		output.Destroy()
		if outputLen == 0 {
			t.Fatalf("legacy model at width %d returned an empty output", width)
		}
		AutoOCRVerification(img)
	}
}
