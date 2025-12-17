//go:build android && arm64

package utils

func init() {
	fmt.Printf("触发GOOS:%v  GOARCH\n\n\n", runtime.GOOS, runtime.GOARCH)
	sharedLibName = "onnxruntime_android_arm64.so"
}
