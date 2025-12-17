//go:build android

package utils

import "fmt"

func init() {
	fmt.Printf("触发GOOS:%v  GOARCH:%v\n\n\n", runtime.GOOS, runtime.GOARCH)
	sharedLibName = "onnxruntime_android_arm64.so"
}
