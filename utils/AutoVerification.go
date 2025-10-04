package utils

import (
	_ "embed"
	"image"
	"log"
	"regexp"
	"strconv"
	"strings"

	"github.com/thedevsaddam/gojsonq"
	ort "github.com/yalue/onnxruntime_go"
)
import "fmt"

// 半自动验证码识别（要自己传入outputShape值）
func SemiOCRVerificationFor(img image.Image, outputShape ort.Shape) string {
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
		fmt.Errorf(err.Error())
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
	img1 := ResizeImage(img, uint(64*img.Bounds().Dx()/img.Bounds().Dy()), 64)
	imgGray := ConvertToGray(img1)
	inputData := ImageToGrayFloatArray(imgGray)
	inputShape := ort.NewShape(1, 1, 64, int64(imgGray.Bounds().Dx()))
	inputTensor, err := ort.NewTensor[float32](inputShape, inputData)
	if err != nil {
		panic(err)
	}

	defer inputTensor.Destroy()
	var outputShape = ort.NewShape(1, 16)
	codeResult := ""
	//失败一次获取输出纬度
	for {
		outputTensor, err := ort.NewEmptyTensor[int64](outputShape)
		defer outputTensor.Destroy()
		session, err := ort.NewAdvancedSession("./assets/common_old.onnx",
			[]string{"input1"}, []string{"output"},
			[]ort.Value{inputTensor}, []ort.Value{outputTensor}, nil)
		defer session.Destroy()
		if err != nil {
			log.Fatal(err)
		}

		err1 := session.Run()
		if err1 != nil {
			if strings.Contains(err1.Error(), "OrtValue shape verification failed. Current shape:") {
				regex := regexp.MustCompile(`Requested shape:{(\d+),(\d+)}`)
				matches := regex.FindStringSubmatch(err1.Error())
				if len(matches) == 3 {
					c1, _ := strconv.Atoi(matches[1])
					c2, _ := strconv.Atoi(matches[2])
					outputShape = ort.NewShape(int64(c1), int64(c2))
					continue
				}
			}
			fmt.Errorf(err1.Error())
			return codeResult
		}

		outputData := outputTensor.GetData()

		for i := 0; i < len(outputData); i++ {
			if outputData[i] != 0 {
				codeResult += gojsonq.New().JSONString(getCharCode()).Find("[" + strconv.Itoa(int(outputData[i])) + "]").(string)
			}
		}
		break
	}
	return codeResult
}
