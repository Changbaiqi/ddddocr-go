package utils

import (
	"errors"
	"image"
	"os"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
)

const betaModelPath = "./assets/common.onnx"
const betaClasses = 8210

// BetaOCRVerification 使用新版 common.onnx + CHARSET_BETA 识别验证码。
//
// 相比旧版 common_old.onnx，新版模型对超星这类字母数字验证码的
// 识别率明显更高；旧版 SemiOCRVerification/AutoOCRVerification 保留
// 用于兼容旧调用。
func BetaOCRVerification(img image.Image) (string, error) {
	if _, err := os.Stat(betaModelPath); err != nil {
		return "", err
	}
	if img == nil || img.Bounds().Dy() <= 0 {
		return "", errors.New("验证码图片高度异常")
	}
	height := 64
	width := height * img.Bounds().Dx() / img.Bounds().Dy()
	if width <= 0 {
		return "", errors.New("验证码图片宽度异常")
	}
	resized := ResizeImage(img, uint(width), uint(height))
	gray := ConvertToGray(resized)
	inputData := ImageToGrayFloatArray(gray)
	inputShape := ort.NewShape(1, 1, int64(height), int64(width))
	input, err := ort.NewTensor[float32](inputShape, inputData)
	if err != nil {
		return "", err
	}
	defer input.Destroy()

	output, err := runCaptchaModel[float32](betaModelPath, "387", input)
	if err != nil {
		return "", err
	}
	defer output.Destroy()

	data := output.GetData()
	shape := output.GetShape()
	if len(shape) != 3 || shape[0] <= 0 || shape[1] != 1 || shape[2] != betaClasses || len(data) == 0 || len(data)%betaClasses != 0 {
		return "", errors.New("新版验证码模型返回了异常的分类维度")
	}

	var builder strings.Builder
	previous := -1
	for base := 0; base < len(data); base += betaClasses {
		best := 0
		bestScore := data[base]
		for class := 1; class < betaClasses; class++ {
			if data[base+class] > bestScore {
				bestScore = data[base+class]
				best = class
			}
		}
		if best != 0 && best != previous && best < len(betaCharset) {
			builder.WriteString(betaCharset[best])
		}
		previous = best
	}
	return builder.String(), nil
}

// 由 ONNX Runtime 按模型的实际输出尺寸分配张量，调用方负责释放。
func runCaptchaModel[T ort.TensorData](modelPath, outputName string, input ort.Value) (*ort.Tensor[T], error) {
	session, err := ort.NewDynamicAdvancedSession(modelPath, []string{"input1"}, []string{outputName}, nil)
	if err != nil {
		return nil, err
	}
	defer session.Destroy()

	outputs := []ort.Value{nil}
	if err := session.Run([]ort.Value{input}, outputs); err != nil {
		if outputs[0] != nil {
			outputs[0].Destroy()
		}
		return nil, err
	}
	if tensor, ok := outputs[0].(*ort.Tensor[T]); ok {
		return tensor, nil
	}
	if outputs[0] != nil {
		outputs[0].Destroy()
	}
	return nil, errors.New("验证码模型返回了异常的输出类型")
}
