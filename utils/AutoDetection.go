package utils

import (
	_ "embed"
	"fmt"
	"image"
	"image/color"
	"log"
	"math"
	"sort"

	"github.com/disintegration/imaging"
	"github.com/fogleman/gg"
	ort "github.com/yalue/onnxruntime_go"
)

// 把 image.Image -> []float32 (CHW, normalized 0..1)
func imageToCHWFloat32(img image.Image, targetW, targetH int) []float32 {
	// resize

	resized := imaging.Resize(img, targetW, targetH, imaging.Lanczos)
	bounds := resized.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	data := make([]float32, 3*h*w)
	idx := 0
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			r, g, b, _ := resized.At(x, y).RGBA()
			rf := float32(r>>8) / 255.0
			gf := float32(g>>8) / 255.0
			bf := float32(b>>8) / 255.0
			// CHW layout: channel major
			data[0*h*w+idx] = rf
			data[1*h*w+idx] = gf
			data[2*h*w+idx] = bf
			idx++
		}
	}
	return data
}

// imageToCHWFloat32Letterbox 把 image.Image 转为 YOLOv12 输入张量
// 输入: targetW, targetH 为模型输入大小（比如 640x640）
// 输出: []float32, [C,H,W] 布局
func ImageToCHWFloat32Letterbox(img image.Image, targetW, targetH int) []float32 {
	// 原图尺寸
	origW := img.Bounds().Dx()
	origH := img.Bounds().Dy()

	// 计算缩放比例
	scale := math.Min(float64(targetW)/float64(origW), float64(targetH)/float64(origH))
	newW := int(float64(origW) * scale)
	newH := int(float64(origH) * scale)

	// 缩放
	resized := imaging.Resize(img, newW, newH, imaging.Linear)

	// 创建目标图（填充黑色）
	dst := imaging.New(targetW, targetH, color.NRGBA{0, 0, 0, 255})
	//dx := (targetW - newW) / 2
	//dy := (targetH - newH) / 2
	//dst = imaging.Paste(dst, resized, image.Pt(dx, dy))
	dst = imaging.Paste(dst, resized, image.Pt(0, 0))

	//err = dc.SavePNG("./assets/output.png")
	//err := gg.SavePNG("./assets/resizeImg.png", dst)
	//if err != nil {
	//	log.Fatal(err)
	//}
	// CHW 数据
	data := make([]float32, 3*targetH*targetW)
	idx := 0
	for y := 0; y < targetH; y++ {
		for x := 0; x < targetW; x++ {
			r, g, b, _ := dst.At(x, y).RGBA()
			rf := float32(r>>8) / 255.0
			gf := float32(g>>8) / 255.0
			bf := float32(b>>8) / 255.0

			// BGR 顺序
			data[0*targetH*targetW+idx] = rf
			data[1*targetH*targetW+idx] = gf
			data[2*targetH*targetW+idx] = bf

			idx++
		}
	}

	return data
}

func AutoDetection(img image.Image, onnxPath string) (string, error) {
	// 初始化环境（如果未初始化）
	if !ort.IsInitialized() {
		if err := ort.InitializeEnvironment(); err != nil {
			return "", err
		}
		defer ort.DestroyEnvironment()
	}

	// 1) 预处理：416x416 RGB CHW float32
	inputW, inputH := 416, 416
	inputData := imageToCHWFloat32(img, inputW, inputH)
	inputShape := ort.NewShape(1, 3, int64(inputH), int64(inputW))
	inputTensor, err := ort.NewTensor[float32](inputShape, inputData)
	if err != nil {
		return "", err
	}
	defer inputTensor.Destroy()

	// 2) 读取模型 input/output 名和 shape
	inputsInfo, outputsInfo, err := ort.GetInputOutputInfo(onnxPath)
	if err != nil {
		return "", err
	}
	if len(inputsInfo) == 0 || len(outputsInfo) == 0 {
		return "", fmt.Errorf("model has no inputs or outputs")
	}
	inputName := inputsInfo[0].Name
	outputName := outputsInfo[0].Name

	// 3) 构造输出 shape（把模型的动态维度替换为合理值）
	outDims := make([]int64, len(outputsInfo[0].Dimensions))
	copy(outDims, outputsInfo[0].Dimensions)

	// 如果维度包含 <=0（dynamic），尝试填入常见 YOLO fallback
	// 如果你的模型不是 YOLO，请根据 InspectModel 的输出手动调整
	needGuess := false
	for _, d := range outDims {
		if d <= 0 {
			needGuess = true
			break
		}
	}
	if needGuess {
		switch len(outDims) {
		case 3:
			// 常见 YOLO (batch, num_boxes, attrs)，对 416 输入常用 num_boxes=10647, attrs=85 (COCO)
			outDims[0] = 1
			outDims[1] = 10647
			outDims[2] = 85
		default:
			// 通用 fallback：把所有 dynamic 维度填成 1
			for i := range outDims {
				if outDims[i] <= 0 {
					outDims[i] = 1
				}
			}
		}
	}

	outputShape := ort.NewShape(outDims...)
	outputTensor, err := ort.NewEmptyTensor[float32](outputShape)
	if err != nil {
		return "", err
	}
	defer outputTensor.Destroy()

	// 4) 创建 session 并运行
	session, err := ort.NewAdvancedSession(onnxPath,
		[]string{inputName}, []string{outputName},
		[]ort.Value{inputTensor}, []ort.Value{outputTensor}, nil)
	if err != nil {
		return "", err
	}
	defer session.Destroy()

	if err := session.Run(); err != nil {
		return "", err
	}

	// 5) 读取输出并简单返回一些调试信息
	data := outputTensor.GetData()   // []float32
	shape := outputTensor.GetShape() // ort.Shape
	// 打印 shape + 前几个值，实际的检测解析（NMS、bbox 解码）需要你按模型格式来实现
	n := len(data)
	limit := 10
	if n < limit {
		limit = n
	}

	// 假设模型输出拿到了 data := out.GetData().([]float32)
	// 输入图片原始宽高
	imgW := img.Bounds().Dx()
	imgH := img.Bounds().Dy()

	detections := ParseDetections(data, imgW, imgH, 0.3)
	fmt.Println(detections)
	// 5. 打印前 20 个框
	numDet := len(data) / 6
	fmt.Println("前 20 个检测框 (raw output):")
	for i := 0; i < 20 && i < numDet; i++ {
		fmt.Printf("[%d] %.2f %.2f %.2f %.2f %.2f %.2f\n",
			i, data[i*6+0], data[i*6+1], data[i*6+2], data[i*6+3], data[i*6+4], data[i*6+5])
	}

	// 6. 绘制到图片
	dc := gg.NewContext(imgW, imgH)
	dc.DrawImage(img, 0, 0)
	for i := 0; i < 10; i++ {
		dc.SetLineWidth(2)
		dc.SetRGBA(1, 0, 0, 0.7)
		dc.DrawRectangle(float64(detections[i].BBox.Dx()), float64(detections[i].BBox.Dy()), float64(detections[i].BBox.Max.X-detections[i].BBox.Min.X), float64(detections[i].BBox.Max.Y-detections[i].BBox.Min.Y))
		dc.Stroke()

		dc.SetRGB(1, 1, 0)
		//dc.DrawStringAnchored(fmt.Sprintf("C%d %.2f", classID, conf), float64(x1), float64(y1)-5, 0, 1)
	}
	// 保存图片
	err = dc.SavePNG("./assets/output.png")
	if err != nil {
		log.Fatal(err)
	}

	return fmt.Sprintf("output shape=%s, dtype=float32, total=%d, first=%v", shape.String(), n, data[:limit]), nil
}
func InspectModel(onnxPath string) error {
	// （可选）初始化环境
	if !ort.IsInitialized() {
		if err := ort.InitializeEnvironment(); err != nil {
			return err
		}
		defer ort.DestroyEnvironment()
	}

	inputs, outputs, err := ort.GetInputOutputInfo(onnxPath)
	if err != nil {
		return err
	}

	fmt.Println("=== Model inputs ===")
	for i, in := range inputs {
		fmt.Printf("[%d] name=%s type=%s dataType=%s dims=%s\n",
			i, in.Name, in.OrtValueType.String(), in.DataType.String(), in.Dimensions.String())
	}

	fmt.Println("=== Model outputs ===")
	for i, out := range outputs {
		fmt.Printf("[%d] name=%s type=%s dataType=%s dims=%s\n",
			i, out.Name, out.OrtValueType.String(), out.DataType.String(), out.Dimensions.String())
	}
	return nil
}

// Detection 结构
type Detection struct {
	BBox     image.Rectangle // 框 (左上,右下)
	Score    float32         // 最终置信度 = conf * class_score
	Class    int             // 类别ID
	Describe string          //描述内容
}

// IoU计算
func IoU(a, b image.Rectangle) float32 {
	inter := a.Intersect(b)
	if inter.Empty() {
		return 0
	}
	interArea := float32(inter.Dx() * inter.Dy())
	aArea := float32(a.Dx() * a.Dy())
	bArea := float32(b.Dx() * b.Dy())
	return interArea / (aArea + bArea - interArea)
}

// NMS实现
func NMS(dets []Detection, iouThresh float32) []Detection {
	sort.Slice(dets, func(i, j int) bool {
		return dets[i].Score > dets[j].Score
	})
	var results []Detection
	used := make([]bool, len(dets))
	for i := 0; i < len(dets); i++ {
		if used[i] {
			continue
		}
		results = append(results, dets[i])
		for j := i + 1; j < len(dets); j++ {
			if used[j] {
				continue
			}
			if IoU(dets[i].BBox, dets[j].BBox) > iouThresh {
				used[j] = true
			}
		}
	}
	return results
}

// 解析 output tensor
func ParseDetections(output []float32, imgW, imgH int, nmsThresh float32) []Detection {
	numDet := len(output) / 6
	var dets []Detection
	for i := 0; i < numDet; i++ {
		cx := output[i*6+0]
		cy := output[i*6+1]
		w := output[i*6+2]
		h := output[i*6+3]

		classScore := output[i*6+5]
		classID := int(classScore) // 如果类ID直接存储在 class_score，可以换成单独字段

		// cx,cy,w,h -> x1,y1,x2,y2
		x1 := (cx - w/2) * 416.0
		y1 := (cy - h/2) * 416.0
		x2 := (cx + w/2) * 416.0
		y2 := (cy + h/2) * 416.0

		bbox := image.Rect(
			int(math.Max(0, float64(x1))),
			int(math.Max(0, float64(y1))),
			int(math.Min(float64(imgW), float64(x2))),
			int(math.Min(float64(imgH), float64(y2))),
		)

		dets = append(dets, Detection{
			BBox:  bbox,
			Score: classScore,
			Class: classID,
		})
	}
	return NMS(dets, nmsThresh)
}
