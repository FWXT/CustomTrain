import os
import re
import json
import streamlit as st
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
# st.set_option('deprecation.showPyplotGlobalUse', False)
import traceback
import time

# 添加CSS样式用于按钮
st.markdown("""
<style>
    /* 全局清除Streamlit生成的空白 */
    /* 这是最重要的一条规则，它将删除所有元素容器间的间距 */
    .element-container {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* 移除所有可能的间隙 */
    div[data-testid="stVerticalBlock"] > div {
        gap: 0px !important;
    }
    
    /* 移除各种特定类的间距 */
    .css-1544g2n, .css-1kyxreq, .css-18e3th9, 
    .css-2trqyj, .css-1d8n9bt, .css-163ttbj,
    .css-1r6slb0, .st-emotion-cache-18ni7ap,
    .st-emotion-cache-zpwto, .st-emotion-cache-16idsys,
    .st-emotion-cache-1wmy9hl, .st-emotion-cache-5rimss {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* 彻底移除空的st.markdown生成的div */
    .stMarkdown div:empty,
    div[data-testid="stMarkdown"]:empty,
    div[data-testid="stMarkdown"] > div:empty,
    div[data-testid="stMarkdown"] > p:empty {
        display: none !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        min-height: 0 !important;
        visibility: hidden !important;
    }
    
    /* 明确移除stMarkdown中所有p标签的边距 */
    div[data-testid="stMarkdown"] p {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* 移除所有stMarkdown的额外空白 */
    div[data-testid="stMarkdown"] {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        min-height: 0 !important;
        line-height: normal !important;
    }
    
    /* 修复streamlit间距的新版类名 */
    .st-emotion-cache-16txtl3, .st-emotion-cache-1629p8f,
    .st-emotion-cache-1avcm0n, .st-emotion-cache-nahz7x,
    .st-emotion-cache-1dp5vir, .st-emotion-cache-1xw8zd0 {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* 顶部和底部导航按钮 */
    .floating-button {
        position: fixed;
        right: 30px;
        z-index: 1000;
        border-radius: 50%;
        width: 45px;
        height: 45px;
        text-align: center;
        line-height: 45px;
        font-size: 20px;
        box-shadow: 2px 2px 8px rgba(0,0,0,0.5);
        cursor: pointer;
        transition: all 0.3s;
        background-color: rgba(60, 60, 60, 0.7);
        color: white;
    }
    .top-button {
        bottom: 90px;
    }
    .bottom-button {
        bottom: 30px;
    }
    .floating-button:hover {
        box-shadow: 3px 3px 10px rgba(0,0,0,0.7);
        transform: scale(1.05);
        background-color: rgba(80, 80, 80, 0.9);
    }

    .json-control-buttons {
        display: flex;
        justify-content: flex-end;
        gap: 10px;
        margin-top: 5px;
    }
    /* 美化JSON控制按钮 */
    .stButton>button {
        border-radius: 4px;
        transition: all 0.2s;
        width: 100%;
        margin-bottom: 8px;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    /* 侧边栏控制按钮样式 */
    .sidebar .stButton>button[data-baseweb="button"]:has(div:contains("上一条")) {
        background-color: #339af0;
        color: white;
    }
    .sidebar .stButton>button[data-baseweb="button"]:has(div:contains("下一条")) {
        background-color: #339af0;
        color: white;
    }
    .sidebar .stButton>button[data-baseweb="button"]:has(div:contains("查看指定JSON")) {
        background-color: #20c997;
        color: white;
        font-weight: bold;
    }
    /* 按钮样式 - 搜索结果中的显示/隐藏按钮 */
    button[data-baseweb="button"]:has(div:contains("显示该JSON")) {
        background-color: #4a4a4a;
        color: white;
    }
    button[data-baseweb="button"]:has(div:contains("隐藏JSON")) {
        background-color: #555555;
        color: white;
    }
    /* 按钮样式 - 内联JSON显示中的上一条/下一条按钮 */
    .inline-json-display button[data-baseweb="button"]:has(div:contains("上一条")) {
        background-color: #4a4a4a;
        color: white;
    }
    .inline-json-display button[data-baseweb="button"]:has(div:contains("下一条")) {
        background-color: #4a4a4a;
        color: white;
    }
    /* 侧边栏分隔线 */
    .sidebar hr {
        margin-top: 1rem;
        margin-bottom: 1rem;
        border: 0;
        border-top: 2px solid rgba(0,0,0,0.1);
    }
    /* 侧边栏标题样式 */
    .sidebar .block-container h2 {
        color: #1e88e5;
        margin-top: 1rem;
        margin-bottom: 1rem;
        font-size: 1.5rem;
    }
    /* 文件信息显示样式 - 支持深色模式 */
    .file-info {
        background-color: rgba(38, 39, 48, 0.2);
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 10px;
        border-left: 4px solid #1e88e5;
        color: inherit;
    }
    .file-info strong {
        color: inherit;
    }
    /* 深色模式下的样式调整 */
    @media (prefers-color-scheme: dark) {
        .file-info {
            background-color: rgba(255, 255, 255, 0.1);
            border-left: 4px solid #64b5f6;
        }
    }
    /* 搜索结果内联JSON显示样式 */
    .inline-json-display {
        margin-left: 20px;
        border-left: 3px solid #666;
        padding-left: 10px;
        margin-bottom: 20px;
    }
    /* 定位锚点样式 */
    #top-anchor {
        position: absolute;
        top: -100px;
        left: 0;
        height: 1px;
        width: 1px;
    }
    #bottom-anchor {
        margin-top: 50px;
        padding-bottom: 100px;
    }

    /* 修复下拉列表中长文件路径的显示问题 */
    .stSelectbox div[data-baseweb="select"] span {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 100%;
        display: inline-block;
    }

    /* 确保下拉列表中的选项显示文件名 */
    .stSelectbox [role="listbox"] [role="option"] {
        display: flex;
        align-items: center;
    }

    .stSelectbox [role="listbox"] [role="option"] div {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 100%;
    }

    /* 悬停时显示完整路径 */
    .stSelectbox [role="listbox"] [role="option"]:hover div {
        overflow: visible;
        white-space: normal;
        word-break: break-all;
        position: relative;
        z-index: 1000;
        background-color: inherit;
    }

    /* 改进的选项卡样式 */
    [data-testid="stTabs"] [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    [data-testid="stTabs"] [data-baseweb="tab"] {
        background-color: rgba(235, 235, 235, 0.6);
        padding: 5px 15px;
        border-radius: 4px 4px 0 0;
        border: 1px solid rgba(200, 200, 200, 0.5);
        border-bottom: none;
    }
    
    [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
        background-color: rgba(255, 255, 255, 0.95);
        border-bottom: 1px solid white;
        margin-bottom: -1px;
        font-weight: bold;
    }
    
    /* 深色模式下的选项卡样式 */
    @media (prefers-color-scheme: dark) {
        [data-testid="stTabs"] [data-baseweb="tab"] {
            background-color: rgba(50, 50, 50, 0.6);
            border: 1px solid rgba(80, 80, 80, 0.5);
            border-bottom: none;
        }
        
        [data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
            background-color: rgba(60, 60, 60, 0.95);
            border-bottom: 1px solid #333;
        }
    }
    
    /* 多行文本显示容器样式 */
    .multiline-text-container {
        margin-bottom: 10px;
        width: 100%;
    }
    
    .multiline-text-container pre {
        width: 100%;
        box-sizing: border-box;
        overflow-y: auto;
    }
    
    /* 多行文本容器的深色模式适配 */
    @media (prefers-color-scheme: dark) {
        .multiline-text-container pre {
            background-color: #1e1e1e !important;
            color: #e0e0e0 !important;
            border: 1px solid #333 !important;
        }
    }
    
    /* 嵌套JSON显示样式 */
    .nested-json {
        margin-left: 15px;
        padding-left: 15px;
        border-left: 3px solid rgba(100, 100, 100, 0.4);
    }
    
    .nested-json-key {
        font-weight: bold;
        color: #4a86e8;
        font-size: 1.05em;
        padding: 2px 4px;
        background-color: rgba(74, 134, 232, 0.08);
        border-radius: 3px;
    }
    
    .nested-json-level-0 {
        border-left-color: #4a86e8;
        border-left-width: 4px;
    }
    
    .nested-json-level-1 {
        border-left-color: #6aa84f;
        border-left-width: 3px;
    }
    
    .nested-json-level-2 {
        border-left-color: #e69138;
    }
    
    .nested-json-level-3 {
        border-left-color: #cc0000;
    }
    
    .nested-json-level-4 {
        border-left-color: #9933cc;
    }
    
    .nested-json-level-5 {
        border-left-color: #3d85c6;
    }
    
    /* 改进文本区域样式，更好的适配深色模式 */
    .stTextArea textarea {
        font-family: 'SFMono-Regular', Consolas, 'Liberation Mono', Menlo, monospace !important;
        font-size: 0.9em !important;
    }
    
    /* 修复深色模式下text_area的可读性 */
    @media (prefers-color-scheme: dark) {
        .stTextArea textarea {
            background-color: rgba(40, 40, 40, 0.8) !important;
            color: #e0e0e0 !important;
        }
    }
    
    /* 调整代码块样式 */
    pre {
        margin-top: 2px !important;
        margin-bottom: 2px !important;
    }
    
    /* 优化tab内容容器 */
    div[data-testid="stTabsContent"] > div[data-baseweb="tab-panel"] {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* 优化嵌套JSON显示 */
    .nested-json {
        margin-left: 10px !important;
        padding-left: 12px !important;
        border-left: 3px solid rgba(100, 100, 100, 0.3) !important;
        margin-top: 4px !important;
        margin-bottom: 4px !important;
    }
    
    /* 美化expander组件 */
    .stExpander {
        border: none !important;
        box-shadow: none !important;
        margin-bottom: 6px !important;
    }
    
    .stExpander > div[data-testid="stExpander"] {
        border: 1px solid rgba(0, 0, 0, 0.05) !important;
        border-radius: 4px !important;
        background-color: rgba(0, 0, 0, 0.01) !important;
    }
    
    .stExpander > div[data-testid="stExpander"] > details {
        background-color: transparent !important;
    }
    
    .stExpander > div[data-testid="stExpander"] > details > summary {
        padding: 3px 8px !important;
        font-size: 0.9em !important;
        background: linear-gradient(to right, rgba(25,118,210,0.04), transparent) !important;
    }
    
    .stExpander > div[data-testid="stExpander"] > details > summary:hover {
        background: linear-gradient(to right, rgba(25,118,210,0.08), transparent) !important;
    }
    
    .stExpander > div[data-testid="stExpander"] > details > summary > span {
        color: #1976d2 !important;
        font-weight: 500 !important;
    }
    
    .stExpander > div[data-testid="stExpander"] > details > div {
        padding-top: 4px !important;
        padding-bottom: 4px !important;
    }
    
    /* 优化列表显示 */
    .compact-list {
        padding: 4px 8px !important;
        background-color: rgba(0, 0, 0, 0.02) !important;
        border-radius: 3px !important;
        display: inline-block !important;
        margin: 2px 0 !important;
    }
    
    /* 深色模式下的expander样式 */
    @media (prefers-color-scheme: dark) {
        .stExpander > div[data-testid="stExpander"] {
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            background-color: rgba(255, 255, 255, 0.01) !important;
        }
        
        .stExpander > div[data-testid="stExpander"] > details > summary {
            background: linear-gradient(to right, rgba(66,165,245,0.04), transparent) !important;
        }
        
        .stExpander > div[data-testid="stExpander"] > details > summary:hover {
            background: linear-gradient(to right, rgba(66,165,245,0.08), transparent) !important;
        }
        
        .stExpander > div[data-testid="stExpander"] > details > summary > span {
            color: #42a5f5 !important;
        }
        
        .nested-json {
            border-left-color: rgba(150, 150, 150, 0.25) !important;
        }
        
        .compact-list {
            background-color: rgba(255, 255, 255, 0.03) !important;
        }
    }

    /* 极小化标签页高度 */
    div[data-testid="stTabs"] div[data-baseweb="tab-list"] {
        min-height: 16px !important;
        padding: 0px !important;
        margin: 0px !important;
        gap: 1px !important;
    }
    
    div[data-testid="stTabs"] button[role="tab"] {
        padding: 0px 5px !important;
        margin: 0px !important;
        line-height: 0.8 !important;
        font-size: 0.65em !important;
        min-height: 16px !important;
        height: 16px !important;
        border-radius: 3px 3px 0 0 !important;
    }
    
    div[data-testid="stTabs"] [data-testid="stTabsContent"] {
        padding-top: 0px !important;
        padding-bottom: 0px !important;
    }
    
    /* 美化JSON key字段 */
    .json-root-key {
        font-weight: 600 !important;
        padding: 2px 8px !important;
        margin: 4px 0px 2px 0px !important;
        border-radius: 3px !important;
        display: inline-block !important;
        background: linear-gradient(135deg, rgba(25,118,210,0.12) 0%, rgba(60,145,230,0.08) 100%) !important;
        border-left: 3px solid #1976d2 !important;
        font-size: 1.05em !important;
        color: #1976d2 !important;
        letter-spacing: 0.3px !important;
    }
    
    .nested-json-key {
        font-weight: 600 !important;
        color: #1976d2 !important;
        background-color: rgba(25,118,210,0.08) !important;
        padding: 2px 5px !important;
        border-radius: 3px !important;
    }
    
    /* 修复text_area空标签警告的样式 */
    .no-label .stTextArea label {
        display: none !important;
        height: 0px !important;
        margin: 0px !important;
        padding: 0px !important;
    }
    
    /* 添加一个轻微的卡片效果，使JSON内容更清晰 */
    div.json-content-wrapper {
        padding: 6px 10px !important;
        margin-bottom: 6px !important;
        border-radius: 4px !important;
        background-color: rgba(0, 0, 0, 0.02) !important;
        border: 1px solid rgba(0, 0, 0, 0.05) !important;
    }
    
    /* 深色模式下的卡片和高亮样式 */
    @media (prefers-color-scheme: dark) {
        .json-root-key {
            background: linear-gradient(135deg, rgba(66,165,245,0.12) 0%, rgba(100,181,246,0.08) 100%) !important;
            border-left: 3px solid #42a5f5 !important;
            color: #42a5f5 !important;
        }
        
        .nested-json-key {
            color: #42a5f5 !important;
            background-color: rgba(66,165,245,0.08) !important;
        }
        
        div.json-content-wrapper {
            background-color: rgba(255, 255, 255, 0.03) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
        }
    }
    
    /* 调整代码块样式 */
    pre {
        margin-top: 0px !important;
        margin-bottom: 0px !important;
    }
    
    /* 减少streamlit组件的默认边距 */
    .element-container {
        margin-top: 0px !important;
        margin-bottom: 0px !important;
        padding-top: 0px !important;
        padding-bottom: 0px !important;
    }
    
    /* 优化JSON显示中的空白 */
    .stJson {
        margin: 0px !important;
        padding: 0px !important;
    }
    
    /* 调整代码块容器的边距 */
    .stCodeBlock {
        margin: 0px !important;
    }
    
    /* 统一标签页内容区域的间距 */
    div[data-testid="stTabs"] [data-testid="stTabsContent"] > div {
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* 移除标签页内容区域的多余空白 */
    div[data-testid="stTabsContent"] > div[data-baseweb="tab-panel"] {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* 特别处理标签页内的json-content-wrapper */
    div[data-testid="stTabsContent"] .json-content-wrapper {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    
    /* 尽可能隐藏textarea的标签 */
    .stTextArea label, .stTextArea div[data-baseweb="form-control"] {
        margin: 0 !important;
        padding: 0 !important;
        min-height: 0 !important;
        line-height: 0 !important;
    }
    
    /* 纯文本和代码块标签间距统一 */
    div[data-baseweb="tab-panel"] > .json-content-wrapper,
    div[data-baseweb="tab-panel"] > .stCodeBlock {
        margin-top: 4px !important;
    }
    
    /* 优化文本区域样式，减少不必要的间距 */
    .stTextArea textarea {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        border: none !important;
        padding-top: 4px !important;
        min-height: 0 !important;
    }
    
    /* 调整代码显示样式，使两个标签页的内容对齐 */
    div.stCodeBlock > div {
        padding-top: 4px !important;
    }
</style>

<div id="top-anchor"></div>
<a href="#top-anchor" class="floating-button top-button">↑</a>
<a href="#bottom-anchor" class="floating-button bottom-button">↓</a>

<script>
// 添加选项卡切换功能
document.addEventListener('DOMContentLoaded', function() {
    function setupTabs() {
        const tabButtons = document.querySelectorAll('.json-tab');
        if (tabButtons.length === 0) return;
        
        tabButtons.forEach(button => {
            button.addEventListener('click', function() {
                const tabsContainer = this.closest('.json-tabs-container');
                const tabId = this.getAttribute('data-tab');
                
                // 移除所有active类
                tabsContainer.querySelectorAll('.json-tab').forEach(b => {
                    b.classList.remove('active');
                });
                tabsContainer.querySelectorAll('.json-tab-content').forEach(c => {
                    c.classList.remove('active');
                });
                
                // 添加active类到当前项
                this.classList.add('active');
                tabsContainer.querySelector(`.json-tab-content[data-tab="${tabId}"]`).classList.add('active');
            });
        });
    }

    // 监听DOM变化，确保在动态添加的内容中也设置选项卡功能
    const observer = new MutationObserver(function(mutations) {
        setupTabs();
    });
    
    setupTabs();
    observer.observe(document.body, { childList: true, subtree: true });
});
</script>
""", unsafe_allow_html=True)

class show_jsonl:
    def __init__(self):
        if hasattr(st.session_state, "JSONL_DIR") and st.session_state["JSONL_DIR"] and \
           hasattr(st.session_state, "jsonl_dir_input") and st.session_state.jsonl_dir_input.split(",") == st.session_state["JSONL_DIR"]:
            return
        st.session_state["JSONL_DIR"] = st.session_state.jsonl_dir_input.split(",") if hasattr(st.session_state, "jsonl_dir_input") else "" # 多个路径用逗号隔开
        st.session_state["search_results"] = []  # 用于保存搜索结果
        st.session_state["time_taken"] = 0
        st.session_state["jsonl_files"] = []  # 用于保存所有的 JSONL 文件路径
        st.session_state["jsonl_files_contents"] = []  # 用于保存所有的 JSONL 文件的内容 # 可能会比较大
        st.session_state["jsonl_files_display"] = []  # 确保初始化显示路径列表
        st.session_state["path_mapping"] = {}  # 确保初始化路径映射
        st.session_state["search_process"] = 0
        st.session_state["search_process_gap"] = 100

        # 初始化嵌套视图首选项
        if "nested_view_preference" not in st.session_state:
            st.session_state["nested_view_preference"] = False
        
        # 初始化编辑状态
        if "editing_json" not in st.session_state:
            st.session_state["editing_json"] = False
        if "edited_data" not in st.session_state:
            st.session_state["edited_data"] = {}

    def load_jsonl_files(self):
        st.session_state["jsonl_files"] = []  # 用于保存所有的 JSONL 文件路径
        st.session_state["jsonl_files_contents"] = []  # 用于保存所有的 JSONL 文件的内容 # 可能会比较大
        st.session_state["jsonl_files_display"] = []  # 用于保存用于显示的相对路径
        st.session_state["path_mapping"] = {}  # 映射相对路径到绝对路径

        for dir_path in st.session_state["JSONL_DIR"]:
            base_dir = os.path.abspath(dir_path)
            for root, _, files in (os.walk(dir_path) if os.path.isdir(dir_path) else [(os.path.dirname(dir_path), "", [os.path.basename(dir_path)])]):
                for file in files:
                    if file.lower().endswith(".jsonl") or file.lower().endswith(".json"):
                        file_path = os.path.join(root, file)
                        # 保存绝对路径用于实际操作
                        st.session_state["jsonl_files"].append(file_path)

                        # 创建相对路径用于显示
                        try:
                            # 获取文件名和相对路径
                            filename = os.path.basename(file_path)
                            rel_path = os.path.relpath(file_path, base_dir)

                            # 如果相对路径太短，就加上最后一级目录名
                            if len(rel_path) < len(filename) + 5:  # 如果相对路径几乎等于文件名
                                parent_dir = os.path.basename(os.path.dirname(file_path))
                                if parent_dir:
                                    rel_path = os.path.join(parent_dir, rel_path)

                            # 创建显示路径，将文件名放在前面，以便在下拉列表中更容易看到
                            display_path = f"{filename} - {os.path.basename(base_dir)}/{os.path.dirname(rel_path)}"
                        except:
                            # 如果出错，使用文件名加父目录
                            display_path = f"{file} - {os.path.basename(os.path.dirname(file_path))}"

                        st.session_state["jsonl_files_display"].append(display_path)
                        st.session_state["path_mapping"][display_path] = file_path

                        # 加载文件内容
                        if file_path.endswith(".jsonl"):
                            with open(file_path, "r", encoding="utf-8") as f:
                                st.session_state["jsonl_files_contents"].append(f.readlines())
                        elif file_path.endswith(".json"):
                            with open(file_path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                                # 如果是数组，转成一行一个JSON
                                if isinstance(data, list):
                                    lines = [json.dumps(item, ensure_ascii=False) + "\n" for item in data]
                                else:
                                    # 如果是单个对象，也放一行
                                    lines = [json.dumps(data, ensure_ascii=False) + "\n"]
                                st.session_state["jsonl_files_contents"].append(lines)
    def show_search_bar(self):
        def clear_checkbox(need_clear_box): # 互斥清除
            st.session_state[need_clear_box] = 0

        # 监听搜索输入框的变化
        def on_search_input_change():
            # 当搜索关键字发生变化时，清除之前查看的指定JSON状态
            if "current_json_file" in st.session_state:
                del st.session_state["current_json_file"]
            if "current_json_row" in st.session_state:
                del st.session_state["current_json_row"]
            st.session_state.editing_json = False # 退出编辑模式

        st.sidebar.subheader("JSONL查看/搜索工具")
        jsonl_dir_input = st.sidebar.text_input("📂 JSONL搜索路径(多个路径请用,隔开)", on_change=lambda:self.__init__(), key = "jsonl_dir_input")
        jsonl_dir_input_button = st.sidebar.button("加载目录", key = "jsonl_dir_input_button")
        if jsonl_dir_input_button:
            self.load_jsonl_files()
            st.session_state["prev_search_query"] = ""

        if not hasattr(st.session_state, "prev_search_query"):
            st.session_state["prev_search_query"] = ""
        if not hasattr(st.session_state, "prev_search_option"):
            st.session_state["prev_search_option"] = (-1, -1, -1, "", "", "")
        search_query = st.sidebar.text_input("🔎 输入搜索关键字", key="search_query", on_change=on_search_input_change)

        # 将搜索选项放在一个扩展器中，使界面更整洁
        with st.sidebar.expander("搜索选项", expanded=False):
            # 基本搜索选项
            case_sensitive = st.checkbox("区分大小写")
            token_match = st.checkbox("全字匹配", on_change = lambda: clear_checkbox("use_regex_button"), key = "token_match_button")
            use_regex = st.checkbox("使用正则表达式", on_change = lambda: clear_checkbox("token_match_button"),  key = "use_regex_button")

            # 添加搜索范围选项
            st.markdown("### 搜索范围和通配符")

            # 添加单独的搜索范围输入框
            file_pattern_input = st.text_input("文件名范围", key="file_pattern_input",
                                          placeholder="例如：test*.jsonl，留空则搜索所有文件")
            path_pattern_input = st.text_input("路径范围", key="path_pattern_input",
                                          placeholder="例如：data/，留空则搜索所有路径")
            key_pattern_input = st.text_input("JSON键范围", key="key_pattern_input",
                                         placeholder="例如：name，留空则搜索所有键")
            st.markdown("""
            支持的通配符和范围语法：
            - `*` - 匹配任意字符（例如：`test*`匹配"test"开头的内容）
            - `?` - 匹配单个字符（例如：`te?t`匹配"test"、"text"等）
            - `file:pattern` - 在指定文件中搜索（例如：`file:test.jsonl`）
            - `path:pattern` - 在指定路径中搜索（例如：`path:data/`）
            - `key:pattern` - 在JSON的指定键中搜索（例如：`key:name`）
            """)
            
        # 添加JSON显示选项
        with st.sidebar.expander("JSON显示选项", expanded=False):
            # 嵌套JSON显示设置
            st.checkbox("是否使用嵌套视图", value=False, key="use_nested_view_global", 
                      help="选择是否使用嵌套视图显示复杂的JSON结构",
                      on_change=lambda: setattr(st.session_state, "nested_view_preference", st.session_state.use_nested_view_global))
            
            st.slider("初始展开层级", min_value=1, max_value=5, value=2, key="initial_expand_level", 
                    help="设置嵌套JSON初始展开的层级数，较大的值会显示更多层级")
            st.slider("大数组/对象显示限制", min_value=10, max_value=100, value=20, key="large_collection_limit",
                    help="设置大型数组或对象最多显示的元素数量")
            
            # 添加使用说明
            st.markdown("""
            ### 嵌套JSON视图说明
            - 点击 **[+]** 或 **[-]** 可以展开或折叠嵌套内容
            - 每个层级使用不同颜色的边框区分
            - 大型数组或对象会被截断显示
            - 过深的嵌套会以代码块方式显示
            """)

        # 添加一条分隔线
        st.sidebar.markdown("---")
        st.sidebar.subheader("直接查看JSON")

        jsonl_select = ""
        display_select = ""
        if jsonl_dir_input:
            # 确保这些键存在于session_state中
            if "jsonl_files" not in st.session_state:
                st.session_state["jsonl_files"] = []
            if "jsonl_files_display" not in st.session_state:
                st.session_state["jsonl_files_display"] = []
            if "path_mapping" not in st.session_state:
                st.session_state["path_mapping"] = {}

            if len(st.session_state["jsonl_files"]) == 0:
                tip = '(路径下无文件或未加载目录)'
            else:
                tip = f'(当前路径: {len(st.session_state["jsonl_files"])} 个文件)'

            # 使用相对路径显示而不是绝对路径
            display_select = st.sidebar.selectbox("📄 选择JSONL文件" + tip, st.session_state["jsonl_files_display"])

            # 转换回绝对路径用于实际操作
            if display_select and display_select in st.session_state["path_mapping"]:
                jsonl_select = st.session_state["path_mapping"][display_select]

        maxDataIndex = 0
        if jsonl_select and st.session_state.get("jsonl_files", []) and st.session_state.get("jsonl_files_contents", []):
            maxDataIndex = len(st.session_state["jsonl_files_contents"][st.session_state["jsonl_files"].index(jsonl_select)])

            # 定义按钮回调函数
            def view_json_callback():
                # 设置当前行数和文件路径
                st.session_state["current_json_file"] = jsonl_select
                st.session_state["current_json_row"] = st.session_state["row_select"]
                # 清除搜索关键字
                if "search_query" in st.session_state:
                    st.session_state["search_query"] = ""
                if "displayed_search_json" in st.session_state:
                    st.session_state["displayed_search_json"] = None
                st.session_state.editing_json = False # 退出编辑模式

            # 使用表单来捕获回车键 - 表单只包含行号输入和查看按钮
            with st.sidebar.form(key="row_select_form"):
                # 从0开始的行号选择器
                st.number_input(
                    f"🔢 选择行号 (0-{maxDataIndex-1})",
                    min_value=0,
                    max_value=maxDataIndex-1,
                    value=0,
                    key="row_select"
                )

                # 将表单提交按钮修改为与查看JSON功能一致的按钮
                st.form_submit_button("👁️ 查看指定JSON",
                                     use_container_width=True,
                                     on_click=view_json_callback)

            # 移除表单外的重复按钮
            # 添加分隔线
            st.sidebar.markdown("---")

        if search_query:
            # 如果有搜索关键字，清除查看指定JSON的状态
            if "current_json_file" in st.session_state:
                del st.session_state["current_json_file"]
            if "current_json_row" in st.session_state:
                del st.session_state["current_json_row"]

            # 获取搜索范围输入框的值
            file_pattern = st.session_state.get("file_pattern_input", "")
            path_pattern = st.session_state.get("path_pattern_input", "")
            key_pattern = st.session_state.get("key_pattern_input", "")

            # 如果搜索查询或选项发生变化，重新执行搜索
            if (st.session_state["prev_search_query"] != search_query or
                st.session_state["prev_search_option"] != (token_match, case_sensitive, use_regex, file_pattern, path_pattern, key_pattern)):
                self.perform_search(search_query, token_match, case_sensitive, use_regex, file_pattern, path_pattern, key_pattern)
                st.session_state.editing_json = False # 退出编辑模式
            self.show_search_result(search_query, token_match, case_sensitive, use_regex)

    def perform_search(self, query, token_match, case_sensitive, use_regex, file_pattern_input="", path_pattern_input="", key_pattern_input=""):
        search_results = []
        t0 = time.time()
        if len(st.session_state.get("jsonl_files", [])) == 0:
            return

        # 解析搜索范围和通配符
        file_pattern = None
        path_pattern = None
        key_pattern = None

        # 优先使用单独输入框中的范围值
        if file_pattern_input:
            # 将通配符转换为正则表达式
            file_pattern = file_pattern_input.replace("*", ".*").replace("?", ".")
        elif "file:" in query:  # 兼容旧的搜索语法
            parts = query.split("file:", 1)
            file_part = parts[1].split()[0] if " " in parts[1] else parts[1]
            # 将通配符转换为正则表达式
            file_pattern = file_part.replace("*", ".*").replace("?", ".")

        if path_pattern_input:
            # 将通配符转换为正则表达式
            path_pattern = path_pattern_input.replace("*", ".*").replace("?", ".")
        elif "path:" in query:  # 兼容旧的搜索语法
            parts = query.split("path:", 1)
            path_part = parts[1].split()[0] if " " in parts[1] else parts[1]
            # 将通配符转换为正则表达式
            path_pattern = path_part.replace("*", ".*").replace("?", ".")

        if key_pattern_input:
            # 将通配符转换为正则表达式
            key_pattern = key_pattern_input.replace("*", ".*").replace("?", ".")
        elif "key:" in query:  # 兼容旧的搜索语法
            parts = query.split("key:", 1)
            key_part = parts[1].split()[0] if " " in parts[1] else parts[1]
            # 将通配符转换为正则表达式
            key_pattern = key_part.replace("*", ".*").replace("?", ".")

        for jsonl_index, jsonl_file in enumerate(st.session_state["jsonl_files"]):
            # 如果有文件模式限制，先检查文件名是否匹配
            if file_pattern:
                filename = os.path.basename(jsonl_file)
                if not re.search(file_pattern, filename, re.IGNORECASE if not case_sensitive else 0):
                    continue

            # 如果有路径模式限制，先检查路径是否匹配
            if path_pattern:
                if not re.search(path_pattern, jsonl_file, re.IGNORECASE if not case_sensitive else 0):
                    continue

            f = st.session_state["jsonl_files_contents"][jsonl_index]
            for line_number, line in enumerate(f):  # 从0开始的行号
                try:
                    # 如果匹配成功，将结果添加到 search_results 中
                    # 如果有key_pattern，需要在JSON的特定键中搜索
                    if key_pattern:
                        try:
                            json_data = json.loads(line)
                            # 找到匹配的键
                            matching_keys = []
                            for k in json_data.keys():
                                if re.match(f"^{key_pattern}$", k, re.IGNORECASE if not case_sensitive else 0):
                                    matching_keys.append(k)

                            if not matching_keys:
                                continue  # 如果没有匹配的键，跳过这一行

                            # 只在匹配的键中搜索
                            for k in matching_keys:
                                search_text = str(json_data[k])
                                ret, content = self.is_match(search_text, query, token_match, case_sensitive, use_regex)
                                if ret:
                                    search_results.append({
                                        "file": jsonl_file,
                                        "line_number": line_number,  # 保存0-based行号
                                        "content": f"{k}: {content}".replace('\n', '\\n'),
                                    })
                                    break  # 找到一个匹配就足够
                        except:
                            # JSON解析失败，继续使用原始文本
                            pass
                    else:
                        # 正常搜索整行文本
                        ret, content = self.is_match(line, query, token_match, case_sensitive, use_regex)
                        if ret:
                            search_results.append({
                                "file": jsonl_file,
                                "line_number": line_number,  # 保存0-based行号
                                "content": content.replace('\n', '\\n'),
                            })
                except:
                    # 忽略无效的 JSON 行
                    print(traceback.format_exc())
                    print("[ERROR] json load failed!", jsonl_file, line_number)

        st.session_state["search_results"] = search_results
        st.session_state["time_taken"] = time.time() - t0
        st.session_state["search_process"] = 0

    def show_search_result(self, _, token_match, case_sensitive, use_regex):
        # 在这里展示搜索结果
        button_key = 0
        if not hasattr(st.session_state, "search_query") or not st.session_state.search_query:
            return
        if hasattr(st.session_state, "search_results"):
            prev_file_name = ""
            if len(st.session_state.search_results) == 0:
                st.write(f"关键字`{st.session_state.search_query}`未找到任何结果")
                return
            else:
                st.session_state["prev_search_query"] = st.session_state.search_query
                # 保存搜索选项和范围
                file_pattern = st.session_state.get("file_pattern_input", "")
                path_pattern = st.session_state.get("path_pattern_input", "")
                key_pattern = st.session_state.get("key_pattern_input", "")
                st.session_state["prev_search_option"] = (token_match, case_sensitive, use_regex, file_pattern, path_pattern, key_pattern)

                min_page = 0
                max_page = len(st.session_state.search_results) // st.session_state["search_process_gap"]
                if len(st.session_state.search_results) % st.session_state["search_process_gap"] == 0: max_page -= 1
                if max_page < 0: max_page = 0
                if min_page != max_page:
                    st.session_state["search_process"] = st.slider(
                        "search_process",
                        min_value=min_page,
                        max_value=max_page,
                        # value=st.session_state["search_process"], # 会有时序
                        label_visibility="collapsed")
                start = st.session_state["search_process"] * st.session_state["search_process_gap"]
                end = min(start + st.session_state["search_process_gap"], len(st.session_state.search_results))
                st.write(f"关键字`{st.session_state.search_query}`找到{len(st.session_state.search_results)}个结果，第{st.session_state['search_process']}页显示[{start},{end-1}]范围，耗时{st.session_state['time_taken']:.4f}s")

            # 添加一个会话状态变量来跟踪当前展示的JSON
            if "displayed_search_json" not in st.session_state:
                st.session_state["displayed_search_json"] = None
                
            # 初始化嵌套视图首选项（如果未初始化）
            if "nested_view_preference" not in st.session_state:
                st.session_state["nested_view_preference"] = False  # 默认关闭嵌套视图

            for i in range(start, end):
                result = st.session_state.search_results[i]
                result_id = f"{result['file']}:{result['line_number']}"

                # 获取用于显示的相对路径
                file_display = result['file']  # 默认使用绝对路径
                # 确保path_mapping存在
                if "path_mapping" in st.session_state:
                    for disp_path, abs_path in st.session_state["path_mapping"].items():
                        if abs_path == result['file']:
                            file_display = disp_path
                            break

                if prev_file_name != result['file']:
                    st.write(f"**文件路径：** {file_display}")
                st.write(f"**行号：** {result['line_number']}  `{result['content']}`")

                # 添加点击显示该文件的功能
                button_key += 1

                # 定义按钮回调函数
                def show_json_callback(result_id=result_id):
                    # 切换显示状态 - 如果当前已经显示这个结果，则隐藏；否则显示这个结果
                    if st.session_state["displayed_search_json"] == result_id:
                        st.session_state["displayed_search_json"] = None
                    else:
                        st.session_state["displayed_search_json"] = result_id

                # 创建显示按钮 - 根据当前是否已经展开来显示不同文本
                button_text = "隐藏JSON" if st.session_state["displayed_search_json"] == result_id else "显示该JSON"
                st.button(button_text, key=f"toggle_json_button_{button_key}",
                          on_click=show_json_callback)

                # 如果当前结果被选中显示，则在此处显示JSON内容
                if st.session_state["displayed_search_json"] == result_id:
                    with st.container():
                        st.markdown('<div class="inline-json-display">', unsafe_allow_html=True)
                        try:
                            # 获取文件内容
                            file_idx = st.session_state["jsonl_files"].index(result['file'])
                            file_contents = st.session_state["jsonl_files_contents"][file_idx]
                            line = file_contents[result['line_number']]  # 使用0-indexed行号
                            json_data = json.loads(line)

                            # 显示文件和行号信息
                            st.info(f"📃 **{file_display} - 第 {result['line_number']} 行**")

                            # 使用回调函数保存用户选择
                            def on_nested_view_search_change():
                                # 保存用户选择到session_state中的全局变量
                                st.session_state["nested_view_preference"] = st.session_state[f"use_nested_view_search_{i}"]
                            
                            # 检查全局嵌套视图首选项
                            use_nested_view = st.session_state["nested_view_preference"]

                            # 移除上一条/下一条按钮，只显示JSON内容
                            for key, value in json_data.items():
                                # 使用特殊样式显示顶层键
                                st.markdown(f"<div class='json-root-key'>{key}</div>", unsafe_allow_html=True)
                                
                                if use_nested_view and isinstance(value, (dict, list)):
                                    # 使用嵌套视图显示复杂类型
                                    unique_id = f"search_json_{i}_{hash(key)}"
                                    self.display_nested_json(value, key=None, level=0, parent_key="", unique_id=unique_id)
                                else:
                                    # 使用带选项卡的代码块显示
                                    code_container = st.container()
                                    container_id = f"code-wrap-container-search-{hash(key)}_{i}" # Unique ID
                                    # 在容器中添加代码块
                                    self.display_tabbed_code(value, container=code_container, unique_id=f"tab_search_{i}_{key}", max_height=400)

                        except Exception as e:
                            st.error(f"加载JSON内容时出错: {str(e)}")

                        st.markdown('</div>', unsafe_allow_html=True)

                prev_file_name = result['file']

    def show_json(self, jsonl_path, row):
        # 保存当前显示的文件和行号 (行号从0开始)
        st.session_state["current_json_file"] = jsonl_path
        st.session_state["current_json_row"] = row

        try:
            file_idx = st.session_state["jsonl_files"].index(jsonl_path)
            f = st.session_state["jsonl_files_contents"][file_idx]
            # 直接使用0-based索引
            line = f[row]
            json_data = json.loads(line)
        except (ValueError, IndexError, json.JSONDecodeError) as e:
            st.error(f"加载或解析 JSON 行时出错: {e}")
            st.error(f"文件: {jsonl_path}, 行号: {row}")
            self.display_multiline_text(line if 'line' in locals() else "无法加载行内容")
            # 清除编辑状态以防万一
            st.session_state.editing_json = False
            return # 无法继续显示或编辑

        # 获取用于显示的相对路径
        display_path = os.path.basename(jsonl_path)  # 默认至少显示文件名
        # 从显示映射中查找相对路径
        if "path_mapping" in st.session_state:
            for disp_path, abs_path in st.session_state["path_mapping"].items():
                if abs_path == jsonl_path:
                    display_path = disp_path
                    break

        # 检查是否在显示搜索结果，只有不在搜索结果模式时才显示控制面板
        if not st.session_state.get("search_query", "") and "current_json_row" in st.session_state and "current_json_file" in st.session_state:
            # 定义回调函数
            def on_prev_click():
                # 更新行号到上一条 (行号从0开始)
                if row > 0:
                    st.session_state["current_json_row"] = row - 1
                    st.session_state.editing_json = False # 切换行时退出编辑

            def on_next_click():
                # 更新行号到下一条 (行号从0开始)
                if row < len(f) - 1:
                    st.session_state["current_json_row"] = row + 1
                    st.session_state.editing_json = False # 切换行时退出编辑

            def on_modify_click():
                st.session_state.editing_json = True
                # 将原始数据存入 edited_data 以便取消
                st.session_state.edited_data = json_data.copy()

            def on_cancel_click():
                st.session_state.editing_json = False
                # 不需要恢复 edited_data，因为下次编辑会重新加载

            def on_save_click():
                try:
                    edited_json = {}
                    # 从 st.text_area 控件收集数据
                    for key in json_data.keys(): # 使用原始数据的键确保顺序和存在性
                        widget_key = f"edit_{key}_{jsonl_path}_{row}"
                        if widget_key in st.session_state:
                             # 尝试将编辑后的文本解析回原始类型（或保持字符串）
                            original_value = json_data[key]
                            edited_text = st.session_state[widget_key]
                            try:
                                # 尝试用原始类型解析，如果失败则保留字符串
                                if isinstance(original_value, (int, float, bool, list, dict)):
                                     # 对布尔值特殊处理
                                    if isinstance(original_value, bool):
                                        if edited_text.lower() == 'true':
                                            edited_json[key] = True
                                        elif edited_text.lower() == 'false':
                                            edited_json[key] = False
                                        else:
                                            raise ValueError("布尔值只能是 True 或 False")
                                    else:
                                        edited_json[key] = json.loads(edited_text)
                                else:
                                    edited_json[key] = edited_text # 保留为字符串
                            except json.JSONDecodeError:
                                # 如果JSON解析失败（比如列表或字典格式不对），直接存为字符串
                                st.warning(f"键 '{key}' 的值无法解析为原始类型，将保存为字符串。原始类型: {type(original_value).__name__}, 输入内容: '{edited_text}'")
                                edited_json[key] = edited_text
                            except ValueError as ve:
                                st.error(f"保存键 '{key}' 时出错: {ve}")
                                return # 保存失败
                        else:
                            # 如果控件状态丢失，则保留原始值？或者报错？
                            st.error(f"无法找到键 '{key}' 的编辑控件状态。")
                            edited_json[key] = json_data[key] # 保留原始值

                    # 转换为 JSON 字符串
                    new_line = json.dumps(edited_json, ensure_ascii=False) + "\n"

                    # 更新内存中的内容
                    st.session_state["jsonl_files_contents"][file_idx][row] = new_line

                    # 写回文件
                    try:
                        with open(jsonl_path, "w", encoding="utf-8") as outfile:
                            outfile.writelines(st.session_state["jsonl_files_contents"][file_idx])
                        st.success(f"已成功保存修改到文件: {display_path} (行号 {row})")
                        st.session_state.editing_json = False
                        # 清理 edited_data
                        st.session_state.edited_data = {}
                    except Exception as write_e:
                        st.error(f"写回文件 {jsonl_path} 时出错: {write_e}")
                        # 可选：如果写入失败，是否回滚内存中的修改？
                        # st.session_state["jsonl_files_contents"][file_idx][row] = line # 恢复原始行
                except Exception as save_e:
                    st.error(f"保存修改时发生错误: {save_e}")
                    st.exception(save_e)

            # 在侧边栏添加控制按钮
            with st.sidebar:
                st.subheader("JSON控制面板")

                # 使用更美观且适合深色模式的方式显示文件信息
                st.markdown(f"""
                <div class="file-info">
                    <div><strong>文件:</strong> {display_path}</div>
                    <div><strong>位置:</strong> 第 <span style="font-weight:bold;color:#ff9800;">{row}</span> 行 / 共 <span style="font-weight:bold;color:#ff9800;">{len(f)}</span> 行</div>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns(2)
                with col1:
                    # 上一条按钮
                    prev_disabled = row <= 0 or st.session_state.editing_json # 编辑时禁用切换
                    st.button("⬆️ 上一条", key=f"prev_json_{row}",
                             disabled=prev_disabled, on_click=on_prev_click, use_container_width=True)

                with col2:
                    # 下一条按钮
                    max_row = len(f)
                    next_disabled = row >= max_row - 1 or st.session_state.editing_json # 编辑时禁用切换
                    st.button("⬇️ 下一条", key=f"next_json_{row}",
                             disabled=next_disabled, on_click=on_next_click, use_container_width=True)

                # 修改和保存按钮
                if st.session_state.editing_json:
                    # 显示保存和取消按钮
                    col_save, col_cancel = st.columns(2)
                    with col_save:
                        st.button("💾 保存修改", key=f"save_json_{row}", on_click=on_save_click, use_container_width=True, type="primary")
                    with col_cancel:
                        st.button("❌ 取消修改", key=f"cancel_json_{row}", on_click=on_cancel_click, use_container_width=True)
                else:
                    # 显示修改按钮
                    st.button("✏️ 修改 JSON", key=f"modify_json_{row}", on_click=on_modify_click, use_container_width=True)

                # 添加嵌套视图选择器
                # 先检查是否有全局性的嵌套视图首选项
                if "nested_view_preference" not in st.session_state:
                    st.session_state["nested_view_preference"] = False  # 默认关闭嵌套视图
                
                # 使用回调函数保存用户选择
                def on_nested_view_change():
                    # 保存用户选择到session_state中的全局变量
                    st.session_state["nested_view_preference"] = st.session_state[f"use_nested_view_{row}"]
                
                # # 显示嵌套视图复选框，使用已保存的值
                # st.checkbox("使用嵌套视图", 
                #            value=st.session_state["nested_view_preference"], 
                #            key=f"use_nested_view_{row}", 
                #            on_change=on_nested_view_change)

            st.write(f"**当前显示：** {display_path} - 第 {row} 行")

            # 显示JSON内容 或 编辑界面
            if st.session_state.editing_json:
                st.info('✏️ 编辑模式：修改下面的值，然后点击侧边栏的"保存修改"。')
                # 使用 st.session_state.edited_data 来填充，防止重置
                current_data = st.session_state.edited_data if st.session_state.edited_data else json_data
                for key, value in current_data.items():
                    # 对于列表或字典，显示JSON字符串形式以便编辑
                    if isinstance(value, (list, dict)):
                        display_value = json.dumps(value, indent=2, ensure_ascii=False)
                        # 计算行数
                        lines = display_value.count('\n') + 1
                        
                        # 使用更精确的高度计算逻辑
                        if lines <= 1:
                            # 单行文本使用较小高度
                            height = 100
                        elif lines <= 5:
                            # 短文本 (2-5行)
                            height = 100 + (lines * 20)
                        elif lines <= 20:
                            # 中等长度文本 (6-20行)
                            height = 200 + ((lines - 5) * 18)
                        else:
                            # 长文本 (>20行) - 更合理的高度增长
                            import math
                            height = 450 + (math.log(lines - 19) * 70)
                        
                        # 确保复杂对象有足够的编辑空间
                        height = max(height, 250)
                        
                        # 限制最大高度
                        height = min(height, 800)
                    else:
                        display_value = str(value)
                        # 计算行数
                        lines = display_value.count('\n') + 1
                        
                        # 使用更精确的高度计算逻辑
                        if lines <= 1:
                            # 单行文本使用较小高度
                            height = 100
                        elif lines <= 5:
                            # 短文本 (2-5行)
                            height = 80 + (lines * 20)
                        elif lines <= 20:
                            # 中等长度文本 (6-20行)
                            height = 180 + ((lines - 5) * 18)
                        else:
                            # 长文本 (>20行)
                            import math
                            height = 400 + (math.log(lines - 19) * 50)
                        
                        # 限制最大高度
                        height = min(height, 800)

                    st.write(f"**{key}:**")
                    # 创建包装容器
                    edit_container = st.container()
                    
                    # 添加包装div来应用样式，避免标签警告
                    edit_container.markdown(
                        f"<div class='code-wrapper' id='code-wrap-container-edit-{hash(key)}_{row}'>", 
                        unsafe_allow_html=True
                    )
                    
                    # 使用非空标签值避免警告
                    edit_container.text_area(
                        label=f"编辑 {key}", # 使用非空标签
                        value=display_value,
                        key=f"edit_{key}_{jsonl_path}_{row}",
                        height=int(height),
                        # 不使用label_visibility="collapsed"，因为我们需要保留CSS选择器
                    )
                    
                    edit_container.markdown("</div>", unsafe_allow_html=True)
            else:
                # 正常显示JSON内容，使用全局嵌套视图首选项
                for key, value in json_data.items():
                    # 使用包装类合并HTML，减少空div
                    combined_html = f"<div class='json-root-key'>{key}</div><div class='json-content-wrapper'>"
                    st.markdown(combined_html, unsafe_allow_html=True)
                    
                    # 使用保存的嵌套视图首选项
                    if st.session_state["nested_view_preference"] and isinstance(value, (dict, list)):
                        # 使用嵌套视图显示复杂类型
                        unique_id = f"json_{row}_{hash(key)}"
                        self.display_nested_json(value, key=None, level=0, parent_key="", unique_id=unique_id)
                    else:
                        # 使用带选项卡的代码块显示
                        code_container = st.container()
                        container_id = f"code-wrap-container-direct-{hash(key)}_{row}" # Unique ID
                        # 在容器中添加代码块
                        self.display_tabbed_code(value, container=code_container, unique_id=f"tab_{row}_{key}", max_height=400)
                    
                    # 添加内容包装器结束
                    st.markdown("</div>", unsafe_allow_html=True)

    def tokenization_text_to_set(self, text, pattern = re.compile(r"[\w_]+", re.ASCII)):
        return { match.group() for match in pattern.finditer(text) }

    def display_multiline_text(self, text, container=None, unique_id=None, max_height=None):
        """
        显示多行文本，并智能调整高度
        
        Args:
            text: 要显示的文本
            container: 可选的 streamlit 容器对象，如果提供则在该容器中显示
            unique_id: 可选的唯一ID，用于确保HTML元素的唯一性
            max_height: 可选的最大高度限制（例如 500 或 "500px"）
        """
        # 生成唯一键
        if unique_id is None:
            import random
            unique_id = f"text_{random.randint(10000, 99999)}_{int(time.time())}"
            
        # 如果是单行文本，尝试删除首尾的引号
        if isinstance(text, str) and text.count('\n') <= 1:
            if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
                text = text[1:-1]
            
        # 处理空字符串的情况
        if text is None or (isinstance(text, str) and text.strip() == ''):
            text = "(空内容)"
        
        # 确保文本是字符串类型
        if not isinstance(text, str):
            text = str(text)
            
        # 计算行数以确定合适的高度
        lines = text.count('\n') + 1
        
        # 根据行数计算基础高度
        if lines <= 1:
            # 单行文本使用较小高度
            base_height = 90
        elif lines <= 5:
            # 短文本 (2-5行) - 每行约18像素
            base_height = 70 + (lines * 18)
        elif lines <= 20:
            # 中等长度文本 (6-20行) - 以稍小的比例增长
            base_height = 160 + ((lines - 5) * 16)
        else:
            # 长文本 (>20行) - 使用对数增长以避免过高
            import math
            base_height = 400 + (math.log(lines - 19) * 40)
            
        # 限制最大高度
        base_height = min(base_height, 700)
        
        # 使用指定的容器或默认流
        target_container = container if container else st
            
        # 如果指定了最大高度，则使用指定值，否则使用计算值
        if max_height:
            # 如果传入的是字符串格式的高度(如"400px")，提取数字部分
            if isinstance(max_height, str):
                if max_height.endswith("px"):
                    try:
                        height = int(max_height[:-2])  # 移除"px"并转换为整数
                    except ValueError:
                        height = base_height  # 转换失败时使用计算值
                else:
                    # 尝试直接将字符串转换为整数
                    try:
                        height = int(max_height)
                    except ValueError:
                        height = base_height
            elif isinstance(max_height, int):
                height = max_height  # 如果已经是整数，直接使用
            else:
                height = base_height  # 其他情况使用计算值
        else:
            height = base_height  # 没有指定使用计算值
        
        # 确保height不小于Streamlit的最小要求(68px)
        height = max(height, 68)
        
        # 确保极短文本不会有过大空间
        if lines <= 3:
            height = min(height, 120)
        
        # 添加自定义CSS来修复text_area标签问题
        target_container.markdown("""
        <style>
        /* 修复text_area空标签警告的样式 */
        .no-label .stTextArea label {
            display: none !important;
            height: 0px !important;
            margin: 0px !important;
            padding: 0px !important;
        }
        
        /* 添加一个轻微的卡片效果，使文本内容更清晰 */
        div.text-content-wrapper {
            padding: 8px 12px !important;
            margin-bottom: 10px !important;
            border-radius: 4px !important;
            background-color: rgba(0, 0, 0, 0.02) !important;
            border: 1px solid rgba(0, 0, 0, 0.05) !important;
        }
        
        /* 深色模式下的卡片样式 */
        @media (prefers-color-scheme: dark) {
            div.text-content-wrapper {
                background-color: rgba(255, 255, 255, 0.03) !important;
                border: 1px solid rgba(255, 255, 255, 0.08) !important;
            }
        }
        </style>
        """, unsafe_allow_html=True)
        
        # 添加一个包装div来应用样式，隐藏标签
        target_container.markdown('<div class="text-content-wrapper">', unsafe_allow_html=True)
        target_container.markdown('<div class="no-label">', unsafe_allow_html=True)
        
        # 使用非空标签来避免警告，然后使用CSS隐藏它
        target_container.text_area(
            label="多行文本内容",  # 提供非空标签避免警告
            value=text,
            height=int(height),
            disabled=True,
            key=f"multiline_text_{unique_id}"
        )
        
        target_container.markdown('</div>', unsafe_allow_html=True)
        target_container.markdown('</div>', unsafe_allow_html=True)

    def display_tabbed_code(self, value, container=None, unique_id=None, code_title="代码块", text_title="纯文本", max_height=None):
        """
        显示带有选项卡的代码块，支持在代码块和纯文本模式之间切换
        使用Streamlit原生UI组件实现选项卡功能
        
        Args:
            value: 要显示的值
            container: 可选的 streamlit 容器对象，如果提供则在该容器中显示
            unique_id: 可选的唯一ID，用于确保HTML元素的唯一性
            code_title: 代码块选项卡的标题
            text_title: 纯文本选项卡的标题
            max_height: 可选的最大高度限制（例如 500）
        """
        # 生成唯一键
        if unique_id is None:
            import random
            unique_id = f"tab_{random.randint(10000, 99999)}_{int(time.time())}"
            
        # 将值转换为适当的格式
        # 将值转换为适当的格式
        if isinstance(value, (dict, list)):
            try:
                code_value = json.dumps(value, indent=2, ensure_ascii=False)
            except:
                code_value = str(value)
        else:
            code_value = str(value)
            code_value = str(value)
            
        # 处理空字符串或空内容的情况
        if not code_value or code_value.strip() == '':
            code_value = "(空内容)"
        
        # 计算适合的高度 - 智能调整高度以适应内容
        # 基于行数和内容长度估算所需空间
        lines = code_value.count('\n') + 1
        
        # 根据行数计算基础高度，使用更精确的算法
        if lines <= 1:
            # 单行文本使用较小高度
            base_height = 90
        elif lines <= 5:
            # 短文本 (2-5行) - 每行约18像素
            base_height = 70 + (lines * 18)
        elif lines <= 20:
            # 中等长度文本 (6-20行) - 以稍小的比例增长
            base_height = 160 + ((lines - 5) * 16)
        else:
            # 长文本 (>20行) - 使用对数增长以避免过高
            import math
            base_height = 400 + (math.log(lines - 19) * 40)
            
        # 限制最大高度
        base_height = min(base_height, 700)
        
        # 如果指定了最大高度，则使用指定值，否则使用计算值
        if max_height:
            # 如果传入的是字符串格式的高度(如"400px")，提取数字部分
            if isinstance(max_height, str):
                if max_height.endswith("px"):
                    try:
                        height = int(max_height[:-2])  # 移除"px"并转换为整数
                    except ValueError:
                        height = base_height  # 转换失败时使用计算值
                else:
                    # 尝试直接将字符串转换为整数
                    try:
                        height = int(max_height)
                    except ValueError:
                        height = base_height
            elif isinstance(max_height, int):
                height = max_height  # 如果已经是整数，直接使用
            else:
                height = base_height  # 其他情况使用计算值
        else:
            height = base_height  # 没有指定使用计算值
        
        # 确保height不小于Streamlit的最小要求(68px)
        height = max(height, 68)
        
        # 确保极短文本不会有过大空间
        if lines <= 3:
            height = min(height, 120)
        
        # CSS注入部分可以使用 markdown_target
        markdown_target = container if container else st
        
        # 添加自定义CSS来极大减小标签页高度，并优化JSON的key显示
        # 使用一次性的markdown调用减少div生成
        markdown_target.markdown("""
        <style>
        /* 极小化标签页高度 */
        div[data-testid="stTabs"] div[data-baseweb="tab-list"] {
            min-height: 16px !important;
            padding: 0px !important;
            margin: 0px !important;
            gap: 1px !important;
        }
        
        div[data-testid="stTabs"] button[role="tab"] {
            padding: 0px 5px !important;
            margin: 0px !important;
            line-height: 0.8 !important;
            font-size: 0.65em !important;
            min-height: 16px !important;
            height: 16px !important;
            border-radius: 3px 3px 0 0 !important;
        }
        
        div[data-testid="stTabs"] [data-testid="stTabsContent"] {
            padding-top: 0px !important;
            padding-bottom: 0px !important;
        }
        
        /* 美化JSON key字段 */
        .json-root-key {
            font-weight: 600 !important;
            padding: 2px 8px !important;
            margin: 4px 0px 2px 0px !important;
            border-radius: 3px !important;
            display: inline-block !important;
            background: linear-gradient(135deg, rgba(25,118,210,0.12) 0%, rgba(60,145,230,0.08) 100%) !important;
            border-left: 3px solid #1976d2 !important;
            font-size: 1.05em !important;
            color: #1976d2 !important;
            letter-spacing: 0.3px !important;
        }
        
        .nested-json-key {
            font-weight: 600 !important;
            color: #1976d2 !important;
            background-color: rgba(25,118,210,0.08) !important;
            padding: 2px 5px !important;
            border-radius: 3px !important;
        }
        
        /* 修复text_area空标签警告的样式 */
        .no-label .stTextArea label {
            display: none !important;
            height: 0px !important;
            margin: 0px !important;
            padding: 0px !important;
        }
        
        /* 添加一个轻微的卡片效果，使JSON内容更清晰 */
        div.json-content-wrapper {
            padding: 6px 10px !important;
            margin-bottom: 6px !important;
            border-radius: 4px !important;
            background-color: rgba(0, 0, 0, 0.02) !important;
            border: 1px solid rgba(0, 0, 0, 0.05) !important;
        }
        
        /* 深色模式下的卡片和高亮样式 */
        @media (prefers-color-scheme: dark) {
            .json-root-key {
                background: linear-gradient(135deg, rgba(66,165,245,0.12) 0%, rgba(100,181,246,0.08) 100%) !important;
                border-left: 3px solid #42a5f5 !important;
                color: #42a5f5 !important;
            }
            
            .nested-json-key {
                color: #42a5f5 !important;
                background-color: rgba(66,165,245,0.08) !important;
            }
            
            div.json-content-wrapper {
                background-color: rgba(255, 255, 255, 0.03) !important;
                border: 1px solid rgba(255, 255, 255, 0.08) !important;
            }
        }
        
        /* 调整代码块样式 */
        pre {
            margin-top: 0px !important;
            margin-bottom: 0px !important;
        }
        
        /* 减少streamlit组件的默认边距 */
        .element-container {
            margin-top: 0px !important;
            margin-bottom: 0px !important;
            padding-top: 0px !important;
            padding-bottom: 0px !important;
        }
        
        /* 优化JSON显示中的空白 */
        .stJson {
            margin: 0px !important;
            padding: 0px !important;
        }
        
        /* 调整代码块容器的边距 */
        .stCodeBlock {
            margin: 0px !important;
        }
        
        /* 统一标签页内容区域的间距 */
        div[data-testid="stTabs"] [data-testid="stTabsContent"] > div {
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }
        
        /* 移除标签页内容区域的多余空白 */
        div[data-testid="stTabsContent"] > div[data-baseweb="tab-panel"] {
            padding: 0 !important;
            margin: 0 !important;
        }
        
        /* 特别处理标签页内的json-content-wrapper */
        div[data-testid="stTabsContent"] .json-content-wrapper {
            margin-top: 0 !important;
            padding-top: 0 !important;
        }
        
        /* 尽可能隐藏textarea的标签 */
        .stTextArea label, .stTextArea div[data-baseweb="form-control"] {
            margin: 0 !important;
            padding: 0 !important;
            min-height: 0 !important;
            line-height: 0 !important;
        }
        
        /* 纯文本和代码块标签间距统一 */
        div[data-baseweb="tab-panel"] > .json-content-wrapper,
        div[data-baseweb="tab-panel"] > .stCodeBlock {
            margin-top: 4px !important;
        }
        
        /* 优化文本区域样式，减少不必要的间距 */
        .stTextArea textarea {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
            border: none !important;
            padding-top: 4px !important;
            min-height: 0 !important;
        }
        
        /* 调整代码显示样式，使两个标签页的内容对齐 */
        div.stCodeBlock > div {
            padding-top: 4px !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # 创建选项卡并填充内容
        if container:
            with container:
                tabs = st.tabs([code_title, text_title])
                self._populate_tabs_content(tabs, value, code_value, lines, height, unique_id)
        else:
            tabs = st.tabs([code_title, text_title])
            self._populate_tabs_content(tabs, value, code_value, lines, height, unique_id)

    # 添加递归处理嵌套JSON的方法
    def display_nested_json(self, value, key=None, level=0, parent_key="", unique_id=""):
        """
        递归地显示嵌套的JSON结构，使其具有更好的可读性和层次感
        
        Args:
            value: 要显示的JSON值
            key: 当前值的键名
            level: 嵌套的层级（用于确定缩进和样式）
            parent_key: 父级键名，用于生成唯一ID
            unique_id: 容器的唯一ID前缀
        """
        # 从session_state获取用户配置，或使用默认值
        max_level = 10  # 最大递归深度
        initial_expand_level = st.session_state.get("initial_expand_level", 2)  # 默认展开前2层
        large_collection_limit = st.session_state.get("large_collection_limit", 20)  # 大集合显示限制
        current_level = min(level, max_level)
        
        # 为当前元素生成唯一ID
        if parent_key and key:
            element_id = f"{unique_id}_{parent_key}_{key}_{level}"
        elif key:
            element_id = f"{unique_id}_{key}_{level}"
        else:
            element_id = f"{unique_id}_{level}_{hash(str(value)) % 10000}"  # 减小哈希值范围，防止ID过长
        
        # 检查是否达到最大递归深度
        if level >= max_level:
            if key is not None:
                st.markdown(f"<span class='nested-json-key'>{key}</span>: <span class='nested-json-value'>[达到最大嵌套深度，以原始格式显示]</span>", unsafe_allow_html=True)
            
            # 使用卡片式容器包装JSON内容
            st.markdown('<div class="json-content-wrapper">', unsafe_allow_html=True)
            
            # 使用多行文本显示替代代码块
            try:
                formatted_value = json.dumps(value, indent=2, ensure_ascii=False)
                self.display_multiline_text(formatted_value, max_height=400)
            except:
                self.display_multiline_text(str(value), max_height=400)
            
            st.markdown('</div>', unsafe_allow_html=True)
            return
        
        # 根据值的类型选择不同的显示方式
        if isinstance(value, dict):
            # 字典类型的显示
            if key is not None:
                # 添加标题的更精确描述
                dict_size = len(value)
                if dict_size == 0:
                    label = f"{key} (空字典)"
                else:
                    label = f"{key} (字典: {dict_size}项)"
                
                # 只在顶层使用expander，避免嵌套
                if level == 0:
                    # 使用Streamlit的expander组件，默认根据level决定是否展开
                    with st.expander(label, expanded=(level < initial_expand_level)):
                        self._display_dict_content(value, level, current_level, large_collection_limit, key, unique_id)
                else:
                    # 对于嵌套层级，使用普通标签和容器
                    st.markdown(f"<div class='nested-json-key'>{label}</div>", unsafe_allow_html=True)
                    # 使用容器包装内容
                    with st.container():
                        self._display_dict_content(value, level, current_level, large_collection_limit, key, unique_id)
            else:
                # 如果没有键名，直接显示内容（根节点）
                # 使用卡片式容器包装全部内容
                st.markdown('<div class="json-content-wrapper">', unsafe_allow_html=True)
                
                # 根据层级添加不同的CSS类
                level_class = f"nested-json-level-{current_level % 6}"
                st.markdown(f"<div class='nested-json {level_class}'>", unsafe_allow_html=True)
                
                # 空字典特殊处理
                if len(value) == 0:
                    st.markdown("<div style='color: #888; font-style: italic; padding: 4px;'>(空字典)</div>", unsafe_allow_html=True)
                else:
                    # 检查字典是否过大
                    if len(value) > large_collection_limit * 2:
                        st.warning(f"该字典包含大量键值对({len(value)}项)，只显示前{large_collection_limit}项")
                        items = list(value.items())[:large_collection_limit]
                        too_large = True
                    else:
                        items = value.items()
                        too_large = False
                    
                    # 递归显示字典中的每个键值对
                    for k, v in items:
                        # 为根节点的每个键添加更明显的分隔
                        st.markdown(f"<div class='json-root-key'>{k}</div>", unsafe_allow_html=True)
                        self.display_nested_json(v, None, level + 1, "root", unique_id)
                    
                    # 如果字典过大，显示省略信息
                    if too_large:
                        st.markdown(f"<div style='color: gray; font-style: italic; padding: 4px;'>... 还有 {len(value) - large_collection_limit} 项未显示</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)
        
        elif isinstance(value, list):
            # 列表类型的显示
            if key is not None:
                # 添加标题的更精确描述
                list_size = len(value)
                if list_size == 0:
                    label = f"{key} (空列表)"
                else:
                    label = f"{key} (列表: {list_size}项)"
                
                # 只在顶层使用expander，避免嵌套
                if level == 0:
                    # 使用Streamlit的expander组件
                    with st.expander(label, expanded=(level < initial_expand_level)):
                        self._display_list_content(value, level, current_level, large_collection_limit, key, unique_id)
                else:
                    # 对于嵌套层级，使用普通标签和容器
                    st.markdown(f"<div class='nested-json-key'>{label}</div>", unsafe_allow_html=True)
                    # 使用容器包装内容
                    with st.container():
                        self._display_list_content(value, level, current_level, large_collection_limit, key, unique_id)
            else:
                # 如果没有键名，直接显示内容
                # 根据层级添加不同的CSS类
                level_class = f"nested-json-level-{current_level % 6}"
                st.markdown(f"<div class='nested-json {level_class}'>", unsafe_allow_html=True)
                
                # 检查列表是否过大
                if len(value) > large_collection_limit * 2:
                    st.warning(f"该列表包含大量元素({len(value)}项)，只显示前{large_collection_limit}项")
                    items = value[:large_collection_limit]
                    too_large = True
                else:
                    items = value
                    too_large = False
                
                # 空列表特殊处理
                if len(items) == 0:
                    st.markdown("<div style='color: #888; font-style: italic;'>(空列表)</div>", unsafe_allow_html=True)
                # 列表元素是否都是简单类型
                elif all(not isinstance(item, (dict, list)) for item in items) and len(items) <= 10:
                    # 如果都是简单类型且数量少，则以行内方式显示
                    formatted_items = []
                    for item in items:
                        if isinstance(item, str):
                            # 处理空字符串
                            if item.strip() == '':
                                formatted_items.append('"(空字符串)"')
                            else:
                                formatted_items.append(f'"{item}"')
                        elif item is None:
                            formatted_items.append("null")
                        else:
                            formatted_items.append(str(item))
                    
                    st.markdown(f"<div class='compact-list'>[{', '.join(formatted_items)}]</div>", unsafe_allow_html=True)
                else:
                    # 递归显示列表中的每个元素
                    for i, item in enumerate(items):
                        # 使用索引作为"键"
                        self.display_nested_json(item, f"[{i}]", level + 1, "root", unique_id)
                
                # 如果列表过大，显示省略信息
                if too_large:
                    st.markdown(f"<div style='color: gray; font-style: italic;'>... 还有 {len(value) - large_collection_limit} 项未显示</div>", unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
        
        else:
            # 基本类型（字符串、数字、布尔值等）的显示
            # 根据类型选择不同的格式化方式
            if isinstance(value, str):
                # 不再截断字符串，显示完整内容
                # 处理空字符串
                if value.strip() == '':
                    formatted_value = '"(空字符串)"'
                else:
                    formatted_value = f'"{value}"'
            elif value is None:
                formatted_value = "null"
            elif isinstance(value, bool):
                formatted_value = "true" if value else "false"
            else:
                formatted_value = str(value)
            
            # 显示键值对
            if key is not None:
                # 检查是否是长字符串或多行字符串，如果是则使用选项卡代码块
                if isinstance(value, str) and (len(value) > 200 or '\n' in value):
                    # 使用带有自定义容器的expander来包装长字符串的显示
                    st.markdown(f"<div><span class='nested-json-key'>{key}</span>:</div>", unsafe_allow_html=True)
                    value_container = st.container()
                    # 在容器中使用display_tabbed_code显示
                    self.display_tabbed_code(value, container=value_container, unique_id=f"tab_nested_{element_id}", max_height=200)
                else:
                    st.markdown(f"<div><span class='nested-json-key'>{key}</span>: <span class='nested-json-value'>{formatted_value}</span></div>", unsafe_allow_html=True)
            else:
                # 检查是否是长字符串或多行字符串，如果是则使用选项卡代码块
                if isinstance(value, str) and (len(value) > 200 or '\n' in value):
                    # 直接使用display_tabbed_code显示
                    self.display_tabbed_code(value, unique_id=f"tab_nested_{element_id}", max_height=200)
                else:
                    st.markdown(f"<div><span class='nested-json-value'>{formatted_value}</span></div>", unsafe_allow_html=True)

    def _display_dict_content(self, value, level, current_level, large_collection_limit, parent_key, unique_id):
        """辅助方法：显示字典内容"""
        # 根据层级添加不同的CSS类
        level_class = f"nested-json-level-{current_level % 6}"
        # 使用单个标记减少div生成
        st.markdown(f"<div class='nested-json {level_class}'>", unsafe_allow_html=True)
        
        # 空字典特殊处理
        if len(value) == 0:
            st.markdown("<div style='color: #888; font-style: italic; padding: 4px;'>(空字典)</div>", unsafe_allow_html=True)
        else:
            # 检查字典是否过大
            if len(value) > large_collection_limit * 2:
                st.warning(f"该字典包含大量键值对({len(value)}项)，只显示前{large_collection_limit}项")
                items = list(value.items())[:large_collection_limit]
                too_large = True
            else:
                items = value.items()
                too_large = False
            
            # 递归显示字典中的每个键值对
            for k, v in items:
                self.display_nested_json(v, k, level + 1, parent_key if parent_key else "root", unique_id)
            
            # 如果字典过大，显示省略信息
            if too_large:
                st.markdown(f"<div style='color: gray; font-style: italic; padding: 4px;'>... 还有 {len(value) - large_collection_limit} 项未显示</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    def _display_list_content(self, value, level, current_level, large_collection_limit, parent_key, unique_id):
        """辅助方法：显示列表内容"""
        # 根据层级添加不同的CSS类
        level_class = f"nested-json-level-{current_level % 6}"
        st.markdown(f"<div class='nested-json {level_class}'>", unsafe_allow_html=True)
        
        # 检查列表是否过大
        if len(value) > large_collection_limit * 2:
            st.warning(f"该列表包含大量元素({len(value)}项)，只显示前{large_collection_limit}项")
            items = value[:large_collection_limit]
            too_large = True
        else:
            items = value
            too_large = False
        
        # 空列表特殊处理
        if len(items) == 0:
            st.markdown("<div style='color: #888; font-style: italic; padding: 4px;'>(空列表)</div>", unsafe_allow_html=True)
        # 列表元素是否都是简单类型
        elif all(not isinstance(item, (dict, list)) for item in items) and len(items) <= 10:
            # 如果都是简单类型且数量少，则以行内方式显示
            formatted_items = []
            for item in items:
                if isinstance(item, str):
                    # 处理空字符串
                    if item.strip() == '':
                        formatted_items.append('"(空字符串)"')
                    else:
                        formatted_items.append(f'"{item}"')
                elif item is None:
                    formatted_items.append("null")
                else:
                    formatted_items.append(str(item))
            
            st.markdown(f"<div class='compact-list'>[{', '.join(formatted_items)}]</div>", unsafe_allow_html=True)
        else:
            # 递归显示列表中的每个元素
            for i, item in enumerate(items):
                # 使用索引作为"键"
                self.display_nested_json(item, f"[{i}]", level + 1, parent_key if parent_key else "root", unique_id)
        
        # 如果列表过大，显示省略信息
        if too_large:
            st.markdown(f"<div style='color: gray; font-style: italic;'>... 还有 {len(value) - large_collection_limit} 项未显示</div>", unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

    def is_match(self, line, query, token_match, case_sensitive, use_regex, preview_len = 88):
        # 解析搜索范围和通配符
        file_pattern = None
        path_pattern = None
        key_pattern = None

        # 检查是否有特殊的搜索范围指令
        # 注意：现在大部分范围处理已经移到perform_search函数中
        # 这里保留以兼容旧的搜索语法
        if "file:" in query:
            parts = query.split("file:", 1)
            query = parts[0].strip()
            file_pattern = parts[1].split()[0] if " " in parts[1] else parts[1]

        if "path:" in query:
            parts = query.split("path:", 1)
            query = parts[0].strip()
            path_pattern = parts[1].split()[0] if " " in parts[1] else parts[1]

        if "key:" in query:
            parts = query.split("key:", 1)
            query = parts[0].strip()
            key_pattern = parts[1].split()[0] if " " in parts[1] else parts[1]

        # 如果查询为空，但有范围限制，则匹配所有内容
        if not query and (file_pattern or path_pattern or key_pattern):
            query = ".*" if use_regex else "*"

        # 获取需要搜索的字段，当前是搜索json字典中所有key: values的值
        text = line # "\n\n".join([f"{k}: {v}" for k, v in json_data.items()])

        # 如果有key限制，尝试解析JSON并只搜索指定的key
        if key_pattern:
            try:
                json_data = json.loads(text)
                # 将通配符转换为正则表达式
                if not use_regex:
                    key_regex = key_pattern.replace("*", ".*").replace("?", ".")
                    key_regex = f"^{key_regex}$"
                else:
                    key_regex = key_pattern

                # 找到匹配的键
                matching_keys = []
                for k in json_data.keys():
                    if re.match(key_regex, k, re.IGNORECASE if not case_sensitive else 0):
                        matching_keys.append(k)

                if not matching_keys:
                    return False, ""

                # 只在匹配的键中搜索
                text = "\n".join([f"{k}: {json_data[k]}" for k in matching_keys])
            except:
                # JSON解析失败，继续使用原始文本
                pass

        if case_sensitive:
            text_csed = text
            query_csed = query
        else:
            text_csed = text.lower()
            query_csed = query.lower()

        # 处理通配符（如果不是正则模式）
        if not use_regex and ("*" in query_csed or "?" in query_csed):
            # 将通配符转换为正则表达式
            query_regex = query_csed.replace("*", ".*").replace("?", ".")
            use_regex = True
            query_csed = query_regex

        suf = ""
        prf = ""
        if token_match or use_regex:
            if token_match: pattern = re.compile('\\b'+query_csed+'\\b')
            elif use_regex: pattern = re.compile(query_csed)
            re_ret = pattern.search(text_csed)
            ret = bool(re_ret)
            if not ret:
                return ret, ""
            s, e = re_ret.span()
            q_len = e - s
            if q_len > preview_len: # 毁灭吧
                return ret, "..." + text[s:e] + "..."
            if len(text_csed) < preview_len:
                q_s_idx = 0
                q_e_idx = len(text_csed)
            else:
                gap = ((preview_len-q_len)//2)
                q_s_idx = s - gap
                q_e_idx = e + gap
                if q_s_idx < 0:
                    q_s_idx = 0
                    q_e_idx -= q_s_idx
                else:
                    suf = "..."
                if q_e_idx > len(text_csed):
                    q_e_idx = len(text_csed)
                else:
                    prf = "..."
            return ret, suf + text[q_s_idx:q_e_idx] + prf
        else:
            ret = query_csed in text_csed
            if not ret:
                return ret, ""
            if len(query_csed) > preview_len: # 毁灭吧
                return ret, "..." + query + "..."
            if len(text_csed) < preview_len:
                q_s_idx = 0
                q_e_idx = len(text_csed)
            else:
                gap = ((preview_len-len(query_csed))//2)
                q_s_idx = text_csed.index(query_csed) - gap
                q_e_idx = text_csed.index(query_csed) + len(query_csed) + gap
                if q_s_idx < 0:
                    q_s_idx = 0
                    q_e_idx -= q_s_idx
                else:
                    suf = "..."
                if q_e_idx > len(text_csed):
                    q_e_idx = len(text_csed)
                else:
                    prf = "..."
            return ret, suf + text[q_s_idx:q_e_idx] + prf

    def layout(self):
        # 确保基础session_state变量已初始化
        for key in ["jsonl_files", "jsonl_files_display", "jsonl_files_contents", "path_mapping"]:
            if key not in st.session_state:
                st.session_state[key] = []

        # 首先显示侧边栏的搜索功能
        self.show_search_bar()

        # 检查是否处于编辑模式，如果是，则强制显示当前编辑的条目
        if st.session_state.get("editing_json", False) and "current_json_row" in st.session_state and "current_json_file" in st.session_state:
             if (st.session_state["jsonl_files"] and
                st.session_state["current_json_file"] in st.session_state["jsonl_files"]):
                file_idx = st.session_state["jsonl_files"].index(st.session_state["current_json_file"])
                row = st.session_state["current_json_row"]
                if (st.session_state["jsonl_files_contents"] and
                    file_idx < len(st.session_state["jsonl_files_contents"]) and
                    row >= 0 and row < len(st.session_state["jsonl_files_contents"][file_idx])):
                     self.show_json(st.session_state["current_json_file"], st.session_state["current_json_row"])

        # 搜索模式和查看指定JSON模式互斥（但编辑模式优先）
        elif not st.session_state.get("search_query", "") and "current_json_row" in st.session_state and "current_json_file" in st.session_state:
            # 确认文件仍然存在于加载的列表中
            if (st.session_state["jsonl_files"] and
                st.session_state["current_json_file"] in st.session_state["jsonl_files"]):
                file_idx = st.session_state["jsonl_files"].index(st.session_state["current_json_file"])
                row = st.session_state["current_json_row"]
                if (st.session_state["jsonl_files_contents"] and
                    file_idx < len(st.session_state["jsonl_files_contents"]) and
                    row >= 0 and row < len(st.session_state["jsonl_files_contents"][file_idx])):
                    self.show_json(st.session_state["current_json_file"], st.session_state["current_json_row"])

        # 添加底部标记，确保底部导航按钮有足够的空间
        st.markdown('<div id="bottom-anchor"></div>', unsafe_allow_html=True)

    def _populate_tabs_content(self, tabs_obj, value_param, code_value_param, lines_param, general_height_param, unique_id_param):
        """Helper method to populate content for display_tabbed_code"""
        # Code block tab
        with tabs_obj[0]:
            if isinstance(value_param, (dict, list)):
                st.json(value_param)
            else:
                st.code(code_value_param, line_numbers=True if lines_param > 5 else False)
        
        # Plain text tab
        with tabs_obj[1]:
            # Calculate text_height specifically for this tab
            current_text_height = general_height_param # Default to general_height_param
            if isinstance(code_value_param, str): # Ensure calculations are for strings
                if len(code_value_param) > 10000:
                    current_text_height = min(general_height_param * 1.8, 700)
                elif len(code_value_param) > 2000 or lines_param > 30:
                    current_text_height = min(general_height_param * 1.4, 650)
                else:
                    current_text_height = min(general_height_param * 1.1, 500)
                
                if lines_param > 20 and current_text_height < 350:
                    current_text_height = 350
                
                if lines_param <= 3:
                    current_text_height = min(current_text_height, 120)

                if len(code_value_param) < 1000 and lines_param <= 20:
                    import html
                    safe_text = html.escape(code_value_param)
                    # Ensure text_height here is the calculated current_text_height
                    html_content = f"""
                    <div class="json-content-wrapper" style="padding:0;margin:0;">
                      <pre style="margin:0;padding:6px;white-space:pre-wrap;word-break:break-all;
                      font-family:monospace;font-size:0.9em;overflow-y:auto;max-height:{int(current_text_height)}px;">
                      {safe_text}</pre>
                    </div>
                    """
                    st.markdown(html_content, unsafe_allow_html=True)
                else:
                    st.text_area(
                        label=" ", 
                        value=code_value_param,
                        height=int(current_text_height),
                        disabled=True,
                        label_visibility="collapsed",
                        key=f"text_area_{unique_id_param}"
                    )
            else: # Should not happen if code_value_param is always string, but as a fallback
                st.text_area(
                    label=" ", 
                    value=str(code_value_param), # ensure string
                    height=int(general_height_param), # fallback height
                    disabled=True,
                    label_visibility="collapsed",
                    key=f"text_area_{unique_id_param}"
                )

if __name__ == "__main__":
    import sys
    # 设置默认端口为6056
    if len(sys.argv) == 1:  # 如果没有提供命令行参数
        sys.argv.extend(["--server.port=6056"])
    show_jsonl().layout()