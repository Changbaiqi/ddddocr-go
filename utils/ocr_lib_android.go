//go:build android

package utils

import (
	"embed"
	"runtime"
)

//go:embed assets/onnxruntime_android_arm64.so
//go:embed assets/common_old.onnx
//go:embed assets/common_tencent.onnx
//go:embed assets/calc_det.onnx
var assets embed.FS

func init() {
	switch runtime.GOARCH {
	case "arm64":
		sharedLibName = "onnxruntime_android_arm64.so"
	}
}
