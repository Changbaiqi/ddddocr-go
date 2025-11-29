package utils

import (
	"errors"
	"fmt"
	"image"
	"math"
	"strconv"
	"strings"

	ort "github.com/yalue/onnxruntime_go"
)

// 腾讯点选专用
var calcChars = []string{"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "-", "x", "=", "?"}

// 腾讯点选
func AutoDetectionForCalc(img image.Image, resultNum int /*返回前多少个检测目标*/) ([]Detection, error) {

	// 初始化环境（如果未初始化）
	if !ort.IsInitialized() {
		if err := ort.InitializeEnvironment(); err != nil {
			return nil, err
		}
		defer ort.DestroyEnvironment()
	}

	// 1) 预处理：640x640 RGB CHW float32
	inputW, inputH := 640, 640
	inputData := ImageToCHWFloat32Letterbox(img, inputW, inputH)

	inputShape := ort.NewShape(1, 3, int64(inputH), int64(inputW))
	inputTensor, err := ort.NewTensor[float32](inputShape, inputData)
	if err != nil {
		return nil, err
	}
	defer inputTensor.Destroy()

	// 2) 读取模型 input/output 名和 shape
	inputsInfo, outputsInfo, err := ort.GetInputOutputInfo("./assets/calc_det.onnx")
	if err != nil {
		return nil, err
	}
	if len(inputsInfo) == 0 || len(outputsInfo) == 0 {
		return nil, fmt.Errorf("model has no inputs or outputs")
	}
	inputName := inputsInfo[0].Name
	outputName := outputsInfo[0].Name

	// 3) 构造输出 shape（把模型的动态维度替换为合理值）
	outDims := make([]int64, len(outputsInfo[0].Dimensions))
	copy(outDims, outputsInfo[0].Dimensions)

	outputShape := ort.NewShape(outDims...)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return nil, err
	}
	defer outputTensor.Destroy()

	// 4) 创建 session 并运行
	session, err := ort.NewAdvancedSession("./assets/calc_det.onnx",
		[]string{inputName}, []string{outputName},
		[]ort.Value{inputTensor}, []ort.Value{outputTensor}, nil)
	if err != nil {
		return nil, err
	}
	defer session.Destroy()

	if err := session.Run(); err != nil {
		return nil, err
	}

	// 5) 读取输出并简单返回一些调试信息
	output := outputTensor.GetData() // []float32
	shape := outputTensor.GetShape() // ort.Shape
	numDetections := int(shape[2])

	detections := []Detection{}

	imgW := img.Bounds().Dx()
	imgH := img.Bounds().Dy()

	for i := 0; i < resultNum; i++ {
		x := output[i*numDetections]
		y := output[i*numDetections+1]
		w := output[i*numDetections+2]
		h := output[i*numDetections+3]
		score := output[i*numDetections+4]
		class := int(output[i*numDetections+5])

		scale := float64(max(imgW, imgH)) / 640.0
		w1 := (float64(w) - float64(x)) * scale
		h1 := (float64(h) - float64(y)) * scale
		x1 := (float64(x)) * scale
		y1 := (float64(y)) * scale
		x2 := x1 + w1
		y2 := y1 + h1

		bbox := image.Rect(
			int(math.Max(0, x1)),
			int(math.Max(0, y1)),
			int(math.Min(float64(imgW), x2)),
			int(math.Min(float64(640), y2)),
		)
		detections = append(detections, Detection{
			BBox:     bbox,
			Score:    score,
			Class:    class,
			Describe: calcChars[class],
		})

	}
	return detections, nil
}

// 根据识别内容自动计算结果
func AutoCalc(detections []Detection) (int, error) {
	detStr := ""
	for i, det := range detections {
		fmt.Printf("[%d] x1=%d x2=%d y1=%d y2=%d score=%f classId=%d classTag=%s \n", i, det.BBox.Min.X, det.BBox.Max.X, det.BBox.Min.Y, det.BBox.Max.Y, det.Score, det.Class, det.Describe)
		detStr += det.Describe
	}
	if strings.Contains(detStr, "+") {
		sp1 := strings.Split(detStr, "+")
		if len(sp1) < 2 {
			return 0, errors.New("计算错误")
		}
		var v1 int
		var v2 int
		atoi, err := strconv.Atoi(sp1[0])
		if err != nil {
			return 0, err
		}
		v1 = atoi
		atoi, err = strconv.Atoi(strings.Split(sp1[1], "=")[0])
		if err != nil {
			return 0, err
		}
		v2 = atoi
		return v1 + v2, nil
	} else if strings.Contains(detStr, "-") {
		sp1 := strings.Split(detStr, "-")
		if len(sp1) < 2 {
			return 0, errors.New("计算错误")
		}
		var v1 int
		var v2 int
		atoi, err := strconv.Atoi(sp1[0])
		if err != nil {
			return 0, err
		}
		v1 = atoi
		atoi, err = strconv.Atoi(strings.Split(sp1[1], "=")[0])
		if err != nil {
			return 0, err
		}
		v2 = atoi
		return v1 - v2, nil
	} else if strings.Contains(detStr, "x") {
		sp1 := strings.Split(detStr, "x")
		if len(sp1) < 2 {
			return 0, errors.New("计算错误")
		}
		var v1 int
		var v2 int
		atoi, err := strconv.Atoi(sp1[0])
		if err != nil {
			return 0, err
		}
		v1 = atoi
		atoi, err = strconv.Atoi(strings.Split(sp1[1], "=")[0])
		if err != nil {
			return 0, err
		}
		v2 = atoi
		return v1 * v2, nil
	} else if strings.Contains(detStr, "*") {
		sp1 := strings.Split(detStr, "*")
		if len(sp1) < 2 {
			return 0, errors.New("计算错误")
		}
		var v1 int
		var v2 int
		atoi, err := strconv.Atoi(sp1[0])
		if err != nil {
			return 0, err
		}
		v1 = atoi
		atoi, err = strconv.Atoi(strings.Split(sp1[1], "=")[0])
		if err != nil {
			return 0, err
		}
		v2 = atoi
		return v1 * v2, nil
	} else if strings.Contains(detStr, "/") {
		sp1 := strings.Split(detStr, "/")
		if len(sp1) < 2 {
			return 0, errors.New("计算错误")
		}
		var v1 int
		var v2 int
		atoi, err := strconv.Atoi(sp1[0])
		if err != nil {
			return 0, err
		}
		v1 = atoi
		atoi, err = strconv.Atoi(strings.Split(sp1[1], "=")[0])
		if err != nil {
			return 0, err
		}
		v2 = atoi
		return v1 * v2, nil
	} else if strings.Contains(detStr, "÷") {
		sp1 := strings.Split(detStr, "÷")
		if len(sp1) < 2 {
			return 0, errors.New("计算错误")
		}
		var v1 int
		var v2 int
		atoi, err := strconv.Atoi(sp1[0])
		if err != nil {
			return 0, err
		}
		v1 = atoi
		atoi, err = strconv.Atoi(strings.Split(sp1[1], "=")[0])
		if err != nil {
			return 0, err
		}
		v2 = atoi
		return v1 * v2, nil
	}

	return 0, nil
}
