package utils

import (
	"embed"
	_ "embed"
	"log"
	"os"
	"runtime"

	ort "github.com/yalue/onnxruntime_go"
)

// 首次调用必须要先进行初始化
//
//go:embed assets/onnxruntime.dll
//go:embed assets/onnxruntime_arm64.so
//go:embed assets/onnxruntime_arm64.dylib
//go:embed assets/onnxruntime.so
//go:embed assets/common_old.onnx
//go:embed assets/common_tencent.onnx
var assets embed.FS

// 数据列表
var assetsList = []string{
	getSharedLibPath(), //放这别移动位置
	"common_old.onnx",
	"common_tencent.onnx",
}

func DDDDOcrCoreInit() {
	//检查文件是否已经复制到本地
	for _, fileName := range assetsList {
		exists, _ := PathExists("./assets/" + fileName)
		if !exists {
			writeAssetsToDisk() // 确保文件都加载了
			break
		}

	}

	loadAiEnvironment() //加载AI环境
}

// 将必要文件复制到当前目录下
func writeAssetsToDisk() {
	PathExistForCreate("./assets")

	for _, fileName := range assetsList {
		resource, err1 := assets.ReadFile("assets/" + fileName)
		if err1 != nil {
			log.Println(err1)
		}
		wf_status := os.WriteFile("./assets/"+fileName, resource, 0644)
		if wf_status != nil {
			log.Fatal(wf_status)
		}
	}
}

// 加载AI环境
func loadAiEnvironment() {
	ort.SetSharedLibraryPath("./assets/" + assetsList[0])
	err := ort.InitializeEnvironment()
	if err != nil {
		panic(err)
	}
	//defer ort.DestroyEnvironment()
}

// 根据不同系统加载不同的运行库
func getSharedLibPath() string {
	if runtime.GOOS == "windows" {
		if runtime.GOARCH == "amd64" {
			return "onnxruntime.dll"
		}
	}
	if runtime.GOOS == "darwin" {
		if runtime.GOARCH == "arm64" {
			return "onnxruntime_arm64.dylib"
		}
	}
	if runtime.GOOS == "linux" {
		if runtime.GOARCH == "arm64" {
			return "onnxruntime_arm64.so"
		}
		return "onnxruntime.so"
	}
	panic("Unable to find a version of the onnxruntime library supporting this system.")
}
