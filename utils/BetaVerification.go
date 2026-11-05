package utils

import (
	"errors"
	"image"
	"os"
	"regexp"
	"strconv"
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

	seqLen := int64(width / 8)
	if seqLen < 1 {
		seqLen = 1
	}
	var output *ort.Tensor[float32]
	for attempt := 0; attempt < 4; attempt++ {
		output, err = ort.NewEmptyTensor[float32](ort.NewShape(seqLen, 1, betaClasses))
		if err != nil {
			return "", err
		}
		session, sessionErr := ort.NewAdvancedSession(
			betaModelPath,
			[]string{"input1"},
			[]string{"387"},
			[]ort.Value{input},
			[]ort.Value{output},
			nil,
		)
		if sessionErr != nil {
			output.Destroy()
			return "", sessionErr
		}
		runErr := session.Run()
		session.Destroy()
		if runErr == nil {
			break
		}
		output.Destroy()
		output = nil
		if expected, ok := parseBetaSeqLen(runErr); ok {
			seqLen = expected
			continue
		}
		return "", runErr
	}
	if output == nil {
		return "", errors.New("新版验证码模型未返回结果")
	}
	defer output.Destroy()

	data := output.GetData()
	var builder strings.Builder
	previous := -1
	for step := 0; step < int(seqLen); step++ {
		base := step * betaClasses
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

func parseBetaSeqLen(err error) (int64, bool) {
	if err == nil {
		return 0, false
	}
	matches := regexp.MustCompile(`\{(\d+)(?:,\s*1)?,\s*8210\}`).FindStringSubmatch(err.Error())
	if len(matches) != 2 {
		return 0, false
	}
	value, parseErr := strconv.ParseInt(matches[1], 10, 64)
	if parseErr != nil || value <= 0 {
		return 0, false
	}
	return value, true
}
