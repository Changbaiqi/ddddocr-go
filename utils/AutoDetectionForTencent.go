package utils

import (
	"fmt"
	"image"
	"math"

	ort "github.com/yalue/onnxruntime_go"
)

// 腾讯点选专用
var chars = []string{
	"乘", "仓", "伯", "伴", "佰", "侈", "侧", "倍", "倡", "偿", "傲",
	"充", "册", "冲", "凹", "办", "北", "匙", "半", "卑", "卜", "卞",
	"厂", "参", "变", "哺", "啊", "场", "埃", "城", "埠", "堡", "备",
	"奔", "奥", "安", "宝", "宠", "尘", "尝", "尺", "层", "岔", "岸",
	"崇", "崩", "差", "巴", "布", "帛", "帮", "常", "弛", "彪", "彻",
	"忱", "悲", "惩", "惫", "惭", "愁", "憋", "懊", "成", "扁", "才",
	"扒", "扮", "扳", "把", "报", "抱", "拌", "拔", "拨", "持", "按",
	"捕", "掣", "搬", "摆", "撤", "播", "敖", "敞", "斑", "昂", "暗",
	"本", "材", "杯", "板", "柏", "标", "栢", "案", "步", "氨", "池",
	"沉", "沧", "泊", "波", "泵", "测", "渤", "澄", "澈", "澳", "灿",
	"炽", "焙", "熬", "爱", "版", "猜", "玻", "班", "瓣", "甭", "畅",
	"畴", "白", "百", "皑", "盎", "睬", "碍", "碑", "秤", "程", "稠",
	"笆", "笨", "策", "筹", "箔", "簿", "绊", "绷", "编", "罢", "翅",
	"翱", "背", "胺", "脖", "膊", "臣", "般", "舱", "舶", "芭", "苍",
	"苯", "菜", "菠", "蔼", "薄", "虫", "补", "衬", "袄", "裁", "诚",
	"诧", "豹", "贝", "财", "趁", "跋", "踌", "蹦", "车", "辈", "辨",
	"辩", "辫", "辰", "边", "迟", "逞", "遍", "邦", "部", "郴", "酬",
	"钡", "钵", "铂", "长", "阿", "陈", "隘", "雹", "靶", "鞍", "颁",
	"餐", "饱", "驰", "驳", "骋", "齿",
}

// 腾讯点选
func AutoDetectionForTencent(img image.Image, onnxPath string, resultNum int /*返回前多少个检测目标*/) ([]Detection, error) {

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
	inputsInfo, outputsInfo, err := ort.GetInputOutputInfo(onnxPath)
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
	session, err := ort.NewAdvancedSession(onnxPath,
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
			BBox:  bbox,
			Score: score,
			Class: class,
		})

	}
	return detections, nil
}
