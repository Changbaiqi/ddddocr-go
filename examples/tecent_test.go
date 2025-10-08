package examples

import (
	"fmt"
	"log"
	"testing"

	"github.com/Changbaiqi/ddddocr-go/utils"
	"github.com/fogleman/gg"
)

// 请依次点击：碍 柏 驳 [{"elem_id":1,"type":"DynAnswerType_POS","data":"539,345"},{"elem_id":2,"type":"DynAnswerType_POS","data":"339,43"},{"elem_id":3,"type":"DynAnswerType_POS","data":"235,339"}]
// 测试腾讯点选目标检测
func TestTencent(t *testing.T) {
	utils.DDDDOcrCoreInit()
	err2 := utils.InspectModel("./assets/common_tencent.onnx")
	if err2 != nil {
		t.Error(err2)
	}

	img, err := utils.ReadImg("E:\\GolandProjects\\ddddocr-go\\examples\\cap_union_new_getcapbysig.png")

	if err != nil {
		panic(err)
	}
	detections, err2 := utils.AutoDetectionForTencent(img, 3)
	if err2 != nil {
		t.Error(err2)
	}
	fmt.Println("detection:", detections)
	for i, det := range detections {
		fmt.Printf("[%d] x1=%d x2=%d y1=%d y2=%d score=%f classId=%d classTag=%s \n", i, det.BBox.Min.X, det.BBox.Max.X, det.BBox.Min.Y, det.BBox.Max.Y, det.Score, det.Class, det.Describe)
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
