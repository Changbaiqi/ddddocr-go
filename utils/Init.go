package utils

import (
	"os"

	ort "github.com/yalue/onnxruntime_go"
)

// 由各个平台文件定义
var sharedLibName string
var modelFiles = []string{
	"common_old.onnx",
	"common_tencent.onnx",
	"calc_det.onnx",
}

// 初始化
func DDDDOcrCoreInit() {
	writeAssetsToDisk()
	loadAiEnvironment()
}

// 将 assets/ 目录内容写入 ./assets
func writeAssetsToDisk() {
	os.MkdirAll("./assets", 0755)

	// 写入动态库
	data, _ := assets.ReadFile("assets/" + sharedLibName)
	os.WriteFile("./assets/"+sharedLibName, data, 0644)

	// 写入模型文件
	for _, name := range modelFiles {
		bin, _ := assets.ReadFile("assets/" + name)
		os.WriteFile("./assets/"+name, bin, 0644)
	}
}

// 加载共享库
func loadAiEnvironment() {
	ort.SetSharedLibraryPath("./assets/" + sharedLibName)
	if err := ort.InitializeEnvironment(); err != nil {
		panic(err)
	}
}
