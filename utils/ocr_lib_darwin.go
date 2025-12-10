//go:build darwin

package utils

import (
	"embed"
	"runtime"
)

//go:embed assets/onnxruntime_osx_x86_64_1.20.1.dylib
//go:embed assets/onnxruntime_osx_arm64_1.20.1.dylib
//go:embed assets/common_old.onnx
//go:embed assets/common_tencent.onnx
//go:embed assets/calc_det.onnx
var assets embed.FS

func init() {
	switch runtime.GOARCH {
	case "amd64":
		sharedLibName = "onnxruntime_osx_x86_64_1.20.1.dylib"
	case "arm64":
		sharedLibName = "onnxruntime_osx_arm64_1.20.1.dylib"
	}
}
