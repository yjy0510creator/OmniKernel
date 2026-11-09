"""OmniKernel —— 算子自动生成与跨硬件适配框架。

三层架构：
- spec      算子描述层（operator_specs/*.yaml，单一事实源）
- engine    模板/特化层（string.Template 渲染 + py_compile 语法闸门）
- agent     LLM 增强与编译-测试反馈闭环（可选，未配置 LLM 时自动退化为模板模式）
"""
__version__ = "1.0.0"
