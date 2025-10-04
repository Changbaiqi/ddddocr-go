package examples

import (
	"fmt"
	"log"
	"testing"

	"github.com/Changbaiqi/ddddocr-go/utils"
	"github.com/fogleman/gg"
)

var chars = []string{"乘", "呈", "逞", "才", "场", "扳", "杯", "摆", "鞍", "斥", "懊", "熬", "常", "笆", "卑", "豹", "澄", "长", "部", "郴", "崇", "雹", "冲", "仓", "伴", "彻", "侈", "澈", "宠", "称", "尘", "卜", "崩", "补", "悲", "奔", "迟", "泵", "半", "渤", "城", "般", "北", "驳", "绊", "伯", "宝", "瓣", "泊", "愁", "标", "扮", "灿", "睬", "尝", "背", "遍", "菜", "餐", "斑", "舶", "报", "材", "白", "憋", "碍", "酬", "策", "测", "差", "佰", "备", "蔼", "澳", "饱", "铂", "保", "撑", "本", "步", "边", "布", "焙", "搬", "爱", "臣", "碑", "箔", "惫", "罢", "倡", "蹦", "钵", "充", "钡", "驰", "参", "侧", "册", "匙", "厂", "持", "层", "舱", "变", "绷", "成", "啊", "播", "稗", "暗", "橙", "班", "敖", "畅", "波", "车", "贝", "倍", "凹", "奥", "胺", "衬", "岔", "玻", "昂", "趁", "苍", "弛", "财", "辰", "诧", "诚", "帮", "皑", "苯", "甭", "拌", "炽", "编", "偿", "哺", "埠", "拔", "池", "秤", "程", "堡", "氨", "芭", "辫", "稠", "隘", "傲", "尺", "版", "巴", "拨", "簿", "畴", "帛", "把", "彪", "筹", "扁", "柏", "跋", "陈", "采", "掣", "菠", "卞", "岸", "盎", "安", "翅", "百", "阿", "抱", "踌", "板", "撤", "裁", "辩", "骋", "办", "按", "忱", "猜", "案", "埃", "袄", "膊", "惭", "唱", "敞", "虫", "并", "辨", "捕", "蚕", "邦", "晨", "颁", "承", "沧", "沉", "靶", "辈", "薄", "脖", "翱", "惩", "齿", "扒"}

// 测试腾讯点选目标检测
func TestTencent(t *testing.T) {
	utils.DDDDOcrCoreInit()
	err2 := utils.InspectModel("./assets/common_tencent.onnx")
	if err2 != nil {
		t.Error(err2)
	}
	img, err := utils.ReadImg("E:\\Yatori-Dev\\tencentImg\\隘驰衬_4db7a2ec66e79f5ee285e73cab5a00d3.png")
	if err != nil {
		panic(err)
	}
	detections, err2 := utils.AutoDetectionForTencent(img, "./assets/common_tencent.onnx", 3)
	if err2 != nil {
		t.Error(err2)
	}
	fmt.Println("detection:", detections)
	for i, det := range detections {
		fmt.Printf("[%d] x1=%d x2=%d y1=%d y2=%d score=%f classId=%d classTag=%s \n", i, det.BBox.Min.X, det.BBox.Max.X, det.BBox.Min.Y, det.BBox.Max.Y, det.Score, det.Class, chars[det.Class])
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
