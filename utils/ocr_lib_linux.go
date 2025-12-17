//go:build linux

package utils

import (
	"embed"
	"runtime"
)

//go:embed assets/onnxruntime_android_arm64.so
//go:embed assets/onnxruntime_linux_x64.so.1.22.0
//go:embed assets/onnxruntime_linux_aarch64.so.1.22.0
//go:embed assets/common_old.onnx
//go:embed assets/common_tencent.onnx
//go:embed assets/calc_det.onnx
var assets embed.FS

func init() {
	switch runtime.GOARCH {
	case "amd64":
		sharedLibName = "onnxruntime_linux_x64.so.1.22.0"
	case "arm64":
		if runtime.GOOS == "android" {
			sharedLibName = "onnxruntime_android_arm64.so"
		} else {
			//sharedLibName = "onnxruntime_linux_aarch64.so.1.22.0"
		}
	}
}
