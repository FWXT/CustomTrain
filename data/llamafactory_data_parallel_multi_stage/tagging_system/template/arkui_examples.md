### 1
#### unidiff
```
--- a/entry/src/main/ets/pages/Index.ets
+++ b/entry/src/main/ets/pages/Index.ets
@@ -5,6 +5,9 @@
   build() {
     Column() {
       Text('拖拽我')
+        .fontColor(Color.Red)
+        .fontSize(20)
+        .fontWeight(FontWeight.Bold)
+        .onDragStart((event: DragEvent) => {
+          console.log('拖拽开始')
+        })
     }
   }
 }
```
#### Qwen3 response
```json
{
    "tags": ["文本通用", "拖拽事件"],
    "tag_reasons": {
        "文本通用": "此段代码对Text组件的字体颜色(.fontColor)、字号(.fontSize)和字体粗细(.fontWeight)进行了设置，这些属于文本通用样式的配置，因此该修改属于文本通用类别",
        "拖拽事件": "此段代码为Text组件添加了.onDragStart拖拽开始事件监听器，用于处理拖拽交互，因此该修改属于拖拽事件类别"
    }
}
```

### 2
#### unidiff
```
--- a/entry/src/main/ets/pages/Index.ets
+++ b/entry/src/main/ets/pages/Index.ets
@@ -3,8 +3,19 @@
 @Component
 struct Index {
   @State counter: number = 0
+  @State textContent: string = "初始文本"
+  
+  @Watch('textContent')
+  onTextChange(propName: string) {
+    console.log(`文本内容变更为: ${this.textContent}`)
+  }

   build() {
     Column() {
-      Text('示例文本')
+      Text(this.textContent)
+        .fontFamily('HarmonySans')
+        .decoration({ type: TextDecorationType.Underline, color: Color.Blue })
+      
+      Button('更新文本')
+        .onClick(() => this.textContent = "更新后的文本")
     }
   }
 }
```
#### Qwen3 response
```json
{
    "tags": ["组件内的状态", "状态变量更改通知", "文本与输入", "按钮与选择", "点击事件", "文本通用"],
    "tag_reasons": {
        "组件内的状态": "此段代码新增了 @State 装饰器修饰的 textContent 状态变量，用于管理组件内部的状态数据，所以该修改属于组件内的状态类别",
        "状态变量更改通知": "此段代码使用了 @Watch('textContent') 来监听 textContent 状态变量的变化，并在变化时触发 onTextChange 回调函数，所以该修改属于状态变量更改通知类别",
        "文本与输入": "此段代码修改了 Text 组件的内容，将其由静态字符串改为绑定状态变量 textContent，涉及文本组件的使用，所以该修改属于文本与输入类别",
        "按钮与选择": "此段代码新增了一个 Button 组件，并绑定了点击事件用于更新文本状态，所以该修改属于按钮与选择类别",
        "点击事件": "此段代码为 Button 组件添加了 onClick 点击事件回调，用于在用户点击按钮时更新文本内容，所以该修改属于点击事件类别",
        "文本通用": "此段代码为 Text 组件设置了 .fontFamily 和 .decoration 属性，用于设置字体和下划线样式，所以该修改属于文本通用类别"
    }
}
```