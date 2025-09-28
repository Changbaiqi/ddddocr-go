package examples

import (
	"fmt"
	"github.com/Changbaiqi/ddddocr-go/utils"
	"testing"

	ort "github.com/yalue/onnxruntime_go"
)

func TestCapter(t *testing.T) {
	utils.DDDDOcrCoreInit()

	img, err := utils.ReadImg("./code4ACcE9bF5D4.png")
	if err != nil {
		panic(err)
	}
	verification := utils.AutoVerification(img, ort.NewShape(1, 18))
	fmt.Println("verification:", verification)
}
func TestCapter1(t *testing.T) {
	utils.DDDDOcrCoreInit()
	err2 := utils.InspectModel("./assets/common_det.onnx")
	if err2 != nil {
		t.Error(err2)
	}
	img, err := utils.ReadImg("./img.png")
	if err != nil {
		panic(err)
	}
	//detection := utils.AutoDetection(img, ort.NewShape(1, 18))
	detection, err2 := utils.AutoDetection(img, "./assets/common_det.onnx")
	if err2 != nil {
		t.Error(err2)
	}
	fmt.Println("detection:", detection)
}
