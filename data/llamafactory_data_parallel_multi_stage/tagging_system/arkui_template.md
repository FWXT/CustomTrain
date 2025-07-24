# Instruction
根据下面代码修改的diff内容，判断此次修改符合下面若干Tags类别中的哪些tag类别。

要求以json格式进行输出，并包含tags和tag_reasons这2列信息。

下面是此次代码修改的diff内容，格式为unidiff格式：
```json
{
    "diff": {{DIFF}}
}
```

# Tags

## 组件内的状态

说明：管理UI组件的内部状态数据

代码关键词：@State

## 状态变量更改通知

说明：状态变量发生变化时回调指定函数

代码关键词：@State; @Watch

## 行列与堆叠

说明：UI组件的行列与堆叠布局

代码关键词：Flex; Column; Row; Stack; RelativeContainer; FolderStack

## 栅格与分栏

说明：UI组件的栅格与分栏布局

代码关键词：GridRow; GridCol; ColumnSplit; SplitLayout; FoldSplitContainer; SideBarContainer

## 滚动与滑动

说明：滚动与滑动UI组件管理

代码关键词：List; ListItem; ListItemGroup; Grid; GridItem; Scroll; Swiper; WaterFlow; FlowItem; ScrollBar; Refresh; ComposeListItem; GridObjectSortComponent; SwipeRefresher

## 按钮与选择

说明：按钮与选择UI组件管理

代码关键词：Button; Toggle; Checkbox; CheckboxGroup; CalendarPicker; DatePicker; TextPicker; TimePicker; Radio; Rating; Select; Slider; DownloadFileButton; ProgressButton; SegmentButton; Filter

## 文本与输入

说明：文本与输入UI组件管理

代码关键词：Text; TextArea; TextInput; RichEditor; Search; Span; ImageSpan; ContainerSpan; SymbolSpan; SymbolGlyph; Hyperlink; RichText; SelectionMenu

## 图片与视频

说明：图片与视频UI组件管理

代码关键词：Image; ImageAnimator; Video; ImageAnalyzerConfig

## 空白与分隔

说明：空白与分隔UI组件管理

代码关键词：Blank; Divider

## 菜单

说明：菜单UI组件管理

代码关键词：Menu; MenuItem; MenuItemGroup; ContextMenu

## 动画

说明：动画UI组件管理

代码关键词：animation; animateTo; keyframeAnimateTo; pageTransition; transition; sharedTransition; geometryTransition; motionPath; Particle; animateToImmediately

## 弹窗

说明：弹窗UI组件管理

代码关键词：AlertDialog; ActionSheet; CustomDialog; CalendarPickerDialog; DatePickerDialog; TimePickerDialog; TextPickerDialog; Dialog

## 自定义组件

说明：自定义UI组件管理

代码关键词：aboutToApper; onDidBuild; aboutToDisappear; onPageShow; onPageHide; onBackPress; aboutToReuse; aboutToRecycle; onWillApplyTheme

## 尺寸设置

说明：UI组件尺寸信息设置

代码关键词：width; height; size; padding; margin; safeAreaPadding; layoutWeight; constraintSize

## 位置设置

说明：UI组件位置信息设置

代码关键词：align; direction; position; markAnchor; offset; alignRules

## 边框设置

说明：UI组件边框信息设置

代码关键词：border; borderStyle; borderWidth; borderColor; borderRadius

## 图片边框设置

说明：UI组件图片边框信息设置

代码关键词：borderImage

## 背景设置

说明：UI组件背景设置

代码关键词：background; backgroundColor; backgroundImage; backgroundImageSize; backgroundImagePosition; backgroundBlurStyle; backdropBlur; backgroundEffect; backgroundImageResizable; backgroundBrightness

## 透明度设置

说明：UI组件透明度设置

代码关键词：opacity

## 显隐控制

说明：UI组件可见性设置

代码关键词：visibility

## 禁用控制

说明：UI组件是否禁用设置

代码关键词：enabled

## 组件标识

说明：UI组件标识信息

代码关键词：id; key

## 文本通用

说明：UI组件中文本相关信息设置

代码关键词：fontColor; fontSize; fontStyle; fontWeight; fontFamily; lineHeight; decoration

## 点击事件

说明：UI组件的点击事件

代码关键词：onClick

## 触摸事件

说明：UI组件的触摸事件

代码关键词：onTouch

## 挂载卸载事件

说明：UI组件的挂载、卸载事件

代码关键词：onAttach; onDetach; onAppear; onDisAppear

## 拖拽事件

说明：UI组件的拖拽事件

代码关键词：onDragStart; onDragEnter; onDragMove; onDragLeave; onDrop; onDragEnd; onPreDrag

## 按键事件

说明：UI组件的按键事件

代码关键词：onKeyEvent; onKeyPreIme; onKeyEventDispatch

# Response
输出格式示例：
```json
{
    "tags":["组件内的状态", "尺寸设置"],
    "tag_reasons":{
        "组件内的状态": "此段代码对管理组件内部状态的状态变量 xxx 进行了 xxx 修改，所以该修改属于组件内的状态类别",
        "尺寸设置": "此段代码对 xxx UI组件的尺寸进行了 xxx 设置，所以该修改属于尺寸设置"
    }
}
```
