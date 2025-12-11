//go:build windows

package utils

import (
	"embed"
	"runtime"
)

//go:embed assets/onnxruntime_win_x64_1.22.0.dll
//go:embed assets/onnxruntime_win_x64_1.22.0.dll
//go:embed assets/onnxruntime_win_x86_1.22.0.dll
//go:embed assets/onnxruntime_win_arm64_1.22.0.dll
//go:embed assets/common_old.onnx
//go:embed assets/common_tencent.onnx
//go:embed assets/calc_det.onnx
var assets embed.FS

func init() {
	switch runtime.GOARCH {
	case "amd64":
		sharedLibName = "onnxruntime_win_x64_1.22.0.dll"
	case "386":
		sharedLibName = "onnxruntime_win_x86_1.22.0.dll"
	case "arm64":
		sharedLibName = "onnxruntime_win_arm64_1.22.0.dll"
	}
}
