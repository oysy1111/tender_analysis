import os
import time
import streamlit as st
from docx import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from openai import RateLimitError


def extract_docx_content(doc: Document, keywords: list = None) -> str:
    """
    从docx文件中提取内容，可选按关键字筛选
    
    Args:
        doc: Document对象
        keywords: 筛选关键字列表
        
    Returns:
        提取的文本内容
    """
    paragraphs = [para.text for para in doc.paragraphs]
    
    # 如果有关键字筛选
    if keywords:
        filtered_paragraphs = [
            para for para in paragraphs
            if any(keyword.lower() in para.lower() for keyword in keywords)
        ]
        return "\n".join(filtered_paragraphs)
    
    return "\n".join(paragraphs)


def initialize_deepseek_chain(api_base: str, api_key: str, model: str):
    """
    初始化DeepSeek问答链
    
    Args:
        api_base: API基础URL
        api_key: API密钥
        model: 模型名称
        
    Returns:
        问答链对象
    """
    prompt = ChatPromptTemplate.from_template("""
    你是一位拥有5年以上经验的专业招标文件分析专家，请仔细阅读以下招标文件内容，并提供一个详细、准确且结构化的分析总结。

    招标文件内容：
    {document}

    请根据上述招标文件内容，提供以下信息的详细分析和总结，每个部分都需要具体信息，不能只写标题：
    
    1. 项目基本信息
       - 项目名称：详细全称
       - 项目编号：招标编号或参考号
       - 招标人：招标单位全称及联系方式
       - 招标代理机构：代理机构名称及联系方式（如适用）
       - 项目审批机关：审批部门（如适用）
       - 资金来源：项目资金构成及来源
    
    2. 项目概况
       - 项目规模：项目总体规模、数量、面积等具体数据
       - 建设地点：详细的项目实施地点
       - 计划工期：具体工期时间范围和工期要求
       - 质量标准：质量验收标准和要求
       - 标段划分：标段数量及各标段具体情况（如适用）
    
    3. 招标范围
       - 具体招标内容：详细列出招标涉及的工作内容
       - 主要工作量：各项工作的具体数量或规模
       - 技术标准：采用的主要技术规范和标准
       - 交付要求：成果交付的形式、时间、地点等要求
    
    4. 投标人资格要求
       - 资质条件：所需的具体资质等级和类别
       - 财务要求：财务状况、注册资本、净资产等要求
       - 业绩要求：需要提供的类似项目经验要求
       - 项目经理要求：项目经理资质、经验等要求
       - 其他要求：项目团队、设备、技术等方面的要求
    
    5. 招标文件获取
       - 获取时间：具体的获取时间范围
       - 获取方式：获取文件的具体方式和地点
       - 文件售价：招标文件售价（如适用）
    
    6. 投标文件递交
       - 截止时间：具体的递交截止日期和时间
       - 递交方式：现场递交、邮寄或电子递交方式
       - 递交地址：具体的递交地点和联系人信息
       - 保证金：投标保证金金额及缴纳方式（如适用）
    
    7. 评标方法和标准
       - 评标方法：采用的评标方法（如综合评估法、经评审的最低投标价法等）
       - 评分标准：各评分因素的权重和评分细则
       - 废标条件：可能导致废标的具体情形
    
    8. 合同条款要点
       - 合同形式：合同类型和格式要求
       - 支付方式：付款条件和方式
       - 履约担保：履约保证金要求（如适用）
       - 违约责任：主要违约责任条款
    
    9. 其他重要信息
       - 踏勘现场：现场踏勘安排（如适用）
       - 答疑安排：答疑会时间安排及提问方式（如适用）
       - 分包要求：是否允许分包及分包限制条件
       - 偏差说明：是否允许投标文件存在偏差及偏差范围
       - 其他特殊要求或注意事项

    要求：
    1. 严格按照上述结构和顺序进行组织
    2. 每个部分都要有具体内容，不能只写标题
    3. 信息要准确，直接来源于招标文件内容
    4. 如某部分内容在招标文件中未提及，请注明"文档中未提及"
    5. 使用清晰的分段和列表形式展示信息
    6. 保持专业、严谨的分析风格
    """)

    llm = ChatOpenAI(
        openai_api_base=api_base,
        openai_api_key=api_key,
        model=model,
    )

    return prompt | llm | StrOutputParser()


def docx_qa(doc: Document, qa_chain, keywords: list = None) -> str:
    """
    对docx文件内容进行问答
    
    Args:
        doc: Document对象
        qa_chain: 问答链对象
        keywords: 筛选关键字列表
        
    Returns:
        模型回答结果
    """
    # 提取内容（可能经过筛选）
    document_content = extract_docx_content(doc, keywords)
    
    # 添加重试机制
    max_retries = 3
    retry_delay = 60  # 重试等待时间（秒）
    
    for attempt in range(max_retries):
        try:
            response = qa_chain.invoke({
                "document": document_content
            })
            return response
        except RateLimitError as e:
            st.warning(f"触发速率限制（第{attempt + 1}次）: {str(e)}")
            if attempt < max_retries - 1:  # 不是最后一次尝试
                st.info(f"等待{retry_delay}秒后重试...")
                time.sleep(retry_delay)
            else:
                st.error("已达到最大重试次数，无法完成请求")
                raise e
        except Exception as e:
            error_msg = str(e)
            if "Insufficient Balance" in error_msg or "402" in error_msg:
                st.error("❌ 错误：您的 DeepSeek API 账户余额不足，请充值后重试。")
                st.info("💡 提示：请登录 [DeepSeek 官网](https://www.deepseek.com/) 查看账户余额并充值。")
                raise e
            else:
                st.error(f"发生其他错误: {error_msg}")
                raise e


def main():
    """主函数，运行Streamlit应用程序"""
    st.set_page_config(page_title="招标文件智能分析工具", layout="wide")
    
    st.title("📄 招标文件智能分析工具")
    st.caption("上传招标文件，使用DeepSeek大模型进行智能分析和总结")
    
       # --- Sidebar: 提取选项 ---
    with st.sidebar:
        st.header("⚙️ 提取选项")
        use_keyword_filter = st.checkbox("启用关键字筛选", value=False)
        
        if use_keyword_filter:
            keywords_input = st.text_area("输入关键字（每行一个）", 
                                        "招标\n投标\n项目\n资格\n投标文件\n截止时间\n评标")
            keywords_list = [kw.strip() for kw in keywords_input.split('\n') if kw.strip()]
        else:
            keywords_list = None

    # --- Main: 文件上传与分析 ---
    uploaded_file = st.file_uploader("上传招标文件", type=["docx"])

    # 直接在代码中配置API参数（请替换为您的实际密钥）
    API_BASE = "https://api.deepseek.com/v1"
    API_KEY = "sk-4e9badbed59143cc94b6b8951c5941fa"  # ⚠️ 请在此处填入您的真实API密钥
    MODEL = "deepseek-chat"

    if uploaded_file:
        # 显示文件信息
        st.info(f"已上传文件: {uploaded_file.name}")
        
        # 加载docx文档
        try:
            doc = Document(uploaded_file)
            st.success("文件加载成功")
        except Exception as e:
            st.error(f"文件加载失败: {e}")
            st.stop()
        
        # 初始化问答链
        try:
            qa_chain = initialize_deepseek_chain(api_base=API_BASE, api_key=API_KEY, model=MODEL)
            st.success("模型连接成功")
        except Exception as e:
            st.error(f"模型初始化失败: {e}")
            st.stop()
        
        # 开始分析按钮
        if st.button("🔍 开始分析", type="primary"):
            with st.spinner("正在分析招标文件，请稍候..."):
                start_time = time.time()
                try:
                    # 调用模型进行分析
                    summary = docx_qa(doc, qa_chain, keywords=keywords_list)
                    elapsed_time = time.time() - start_time
                    
                    # 显示结果
                    st.success(f"分析完成，耗时: {elapsed_time:.2f} 秒")
                    st.markdown("## 📊 分析结果")
                    st.markdown(summary)
                    
                except Exception as e:
                    st.error(f"分析过程中出现错误: {e}")
    else:
        st.info("请上传一个.docx格式的招标文件")


if __name__ == "__main__":
    main()