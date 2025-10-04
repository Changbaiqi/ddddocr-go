# dddddocr-go

---

## 🚀 项目说明
* 这是一个ddddocr的移植库，将python的ddddocr移植到go，所以取名为ddddocr-go
* 目前只实现了ddddocr的ocr功能，解决常见的文字，字母，数字识别还是没问题的
* 适配Windows、Linux

## ⚒️ 如何导入最新版本？
```shell
go get github.com/Changbaiqi/ddddocr-go@latest
````

## 🚀 简单使用教程
```go
package main
import (
	"log"
	ddddocr "github.com/Changbaiqi/ddddocr-go/utils"
	ort "github.com/yalue/onnxruntime_go"
)

func main(){
	//初始化ddddocr库，一般全局初始化一次即可
	ddddocr.DDDDOcrCoreInit()
	//读取图片
	img, err := dddddocr.ReadImg("./code4ACcE9bF5D4.png")
	if err != nil {
		panic(err)
	}
	//识别直接返回结果
	verification := ddddocr.AutoOCRVerification(img)
	log.Println("verification:", verification)
}
```

## ⚒️ 可能出现的问题
如果报错`Platform-specific initialization failed`，可以参考以下文档：
[Platform-specific initialization failed](https://yatori-dev.github.io/yatori-docs/yatori-go-console/docs.html#%E2%9D%93-%E5%85%B3%E4%BA%8E%E6%8A%A5%E9%94%99platform-specific-initialization-failed%E8%A7%A3%E5%86%B3)