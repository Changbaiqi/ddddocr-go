package utils

import (
	"fmt"
	"image"
	"math"

	ort "github.com/yalue/onnxruntime_go"
)

// 腾讯点选专用
var chars = []string{"乘", "呈", "逞", "才", "场", "扳", "杯", "摆", "鞍", "斥", "懊", "熬", "常", "笆", "卑", "豹", "澄", "长", "部", "郴", "崇", "雹", "冲", "仓", "伴", "彻", "侈", "澈", "宠", "称", "尘", "卜", "崩", "补", "悲", "奔", "迟", "泵", "半", "渤", "城", "般", "北", "驳", "绊", "伯", "宝", "瓣", "泊", "愁", "标", "扮", "灿", "睬", "尝", "背", "遍", "菜", "餐", "斑", "舶", "报", "材", "白", "憋", "碍", "酬", "策", "测", "差", "佰", "备", "蔼", "澳", "饱", "铂", "保", "撑", "本", "步", "边", "布", "焙", "搬", "爱", "臣", "碑", "箔", "惫", "罢", "倡", "蹦", "钵", "充", "钡", "驰", "参", "侧", "册", "匙", "厂", "持", "层", "舱", "变", "绷", "成", "啊", "播", "稗", "暗", "橙", "班", "敖", "畅", "波", "车", "贝", "倍", "凹", "奥", "胺", "衬", "岔", "玻", "昂", "趁", "苍", "弛", "财", "辰", "诧", "诚", "帮", "皑", "苯", "甭", "拌", "炽", "编", "偿", "哺", "埠", "拔", "池", "秤", "程", "堡", "氨", "芭", "辫", "稠", "隘", "傲", "尺", "版", "巴", "拨", "簿", "畴", "帛", "把", "彪", "筹", "扁", "柏", "跋", "陈", "采", "掣", "菠", "卞", "岸", "盎", "安", "翅", "百", "阿", "抱", "踌", "板", "撤", "裁", "辩", "骋", "办", "按", "忱", "猜", "案", "埃", "袄", "膊", "惭", "唱", "敞", "虫", "并", "辨", "捕", "蚕", "邦", "晨", "颁", "承", "沧", "沉", "靶", "辈", "薄", "脖", "翱", "惩", "齿", "扒"}

// 腾讯点选
func AutoDetectionForTencent(img image.Image, resultNum int /*返回前多少个检测目标*/) ([]Detection, error) {

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
	inputsInfo, outputsInfo, err := ort.GetInputOutputInfo("./assets/common_tencent.onnx")
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
	session, err := ort.NewAdvancedSession("./assets/common_tencent.onnx",
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
			Describe: chars[class],
		})

	}
	return detections, nil
}
