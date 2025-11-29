package examples

import (
	"fmt"
	"log"
	"sort"
	"testing"

	"github.com/Changbaiqi/ddddocr-go/utils"
	"github.com/fogleman/gg"
)

func TestCalcDet(t *testing.T) {
	utils.DDDDOcrCoreInit()
	err2 := utils.InspectModel("./assets/calc_det.onnx")
	if err2 != nil {
		t.Error(err2)
	}

	img, err := utils.ReadImg("E:\\Yatori-Dev\\yatori-go-core\\examples\\qsxt_code\\qsxt_code_13.png")

	if err != nil {
		panic(err)
	}
	detections, err2 := utils.AutoDetectionForCalc(img, 7)
	if err2 != nil {
		t.Error(err2)
	}
	fmt.Println("detection:", detections)
	sort.Slice(detections, func(i, j int) bool {
		if detections[i].BBox.Min.X < detections[j].BBox.Min.X {
			return true
		}
		return false
	})

	calc, err2 := utils.AutoCalc(detections)
	if err2 != nil {
		fmt.Println(err2)
	} else {
		fmt.Println("计算结果：", calc)
	}

	imgW := img.Bounds().Dx()
	imgH := img.Bounds().Dy()
	// 6. 绘制到图片
	dc := gg.NewContext(imgW, imgH)
	dc.DrawImage(img, 0, 0)

	for _, det := range detections {
		dc.DrawRectangle(float64(det.BBox.Min.X), float64(det.BBox.Min.Y), float64(det.BBox.Max.X-det.BBox.Min.X), float64(det.BBox.Max.Y-det.BBox.Min.Y))
		dc.Stroke()
		dc.SetRGB(1, 1, 0)
	}

	// 保存图片
	err = dc.SavePNG("./assets/output.png")
	if err != nil {
		log.Fatal(err)
	}

}
