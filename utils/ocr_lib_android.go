//go:build android && arm64

package utils

func init() {
	sharedLibName = "onnxruntime_android_arm64.so"
}
