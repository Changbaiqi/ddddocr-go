package utils

import (
	_ "embed"
	"image"
	"log"
	"strconv"
	"strings"

	"github.com/thedevsaddam/gojsonq"
	ort "github.com/yalue/onnxruntime_go"
)

// 半自动验证码识别（要自己传入outputShape值）
func SemiOCRVerification(img image.Image, outputShape ort.Shape) string {
	img1 := ResizeImage(img, uint(64*img.Bounds().Dx()/img.Bounds().Dy()), 64)
	imgGray := ConvertToGray(img1)

	inputData := ImageToGrayFloatArray(imgGray)
	inputShape := ort.NewShape(1, 1, 64, int64(imgGray.Bounds().Dx()))
	inputTensor, err := ort.NewTensor[float32](inputShape, inputData)

	if err != nil {
		panic(err)
	}

	defer inputTensor.Destroy()
	// This hypothetical network maps a 2x5 input -> 2x3x4 output.
	outputTensor, err := ort.NewEmptyTensor[int64](outputShape)
	defer outputTensor.Destroy()
	session, err := ort.NewAdvancedSession("./assets/common_old.onnx",
		[]string{"input1"}, []string{"output"},
		[]ort.Value{inputTensor}, []ort.Value{outputTensor}, nil)
	defer session.Destroy()
	if err != nil {
		log.Fatal(err)
	}

	err = session.Run()
	if err != nil {
		return ""
	}

	outputData := outputTensor.GetData()
	codeResult := ""
	for i := 0; i < len(outputData); i++ {
		if outputData[i] != 0 {
			codeResult += gojsonq.New().JSONString(getCharCode()).Find("[" + strconv.Itoa(int(outputData[i])) + "]").(string)
		}
	}
	return codeResult
}

// 全自动
func AutoOCRVerification(img image.Image) string {
	if img == nil {
		return ""
	}
	height := img.Bounds().Dy()
	if height <= 0 {
		return ""
	}
	width := 64 * img.Bounds().Dx() / height
	if width <= 0 {
		return ""
	}
	resized := ResizeImage(img, uint(width), 64)
	gray := ConvertToGray(resized)
	input, err := ort.NewTensor[float32](ort.NewShape(1, 1, 64, int64(width)), ImageToGrayFloatArray(gray))
	if err != nil {
		return ""
	}
	defer input.Destroy()

	output, err := runCaptchaModel[int64]("./assets/common_old.onnx", "output", input)
	if err != nil {
		return ""
	}
	defer output.Destroy()
	shape := output.GetShape()
	if len(shape) != 2 || shape[0] != 1 || shape[1] <= 0 {
		return ""
	}

	var code strings.Builder
	for _, id := range output.GetData() {
		if id != 0 {
			char, ok := gojsonq.New().JSONString(getCharCode()).Find("[" + strconv.FormatInt(id, 10) + "]").(string)
			if !ok {
				return ""
			}
			code.WriteString(char)
		}
	}
	return code.String()
}
