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
       - 项目名称：详细全称（例如：大唐江西抚州 2×1000MW 煤电扩建项目主体施工 - 一标段）
       - 项目编号：招标编号或参考号（例如：CWEME-202505JXFZ-S001）
       - 招标人：招标单位全称及联系方式（例如：大唐抚州第二发电有限公司）
       - 招标代理机构：代理机构名称及联系方式（例如：北京国电工程招标有限公司）
       - 项目审批机关：审批部门（如适用，文档中未提及时注明）
       - 资金来源：项目资金构成及来源（文档中未提及时注明）
       - 建设地点：详细的项目实施地点（例如：江西省抚州市临川区鹏田乡、青泥镇）
       - 建设规模：项目总体规模、容量等具体数据（例如：2000MW，2×1000MW）
       - 计划工期：总工期和关键节点工期（例如：总工期955天，从2025年6月20日至2028年1月31日；3号机组投产时间2027年7月30日）
       - 授标原则：标段划分和中标规则（例如：共2个标段，投标人最多中标1个标段；若2个标段均排名第一，优先授予一标段）
    
    2. 项目概况
       - 项目规模：具体数据如装机容量、单位数量（例如：2×1000MW机组）
       - 建设地点：详细地址（例如：江西省抚州市临川区鹏田乡、青泥镇）
       - 计划工期：开始和结束日期，关键里程碑（例如：总工期955天，3号机组施工770天）
       - 质量标准：质量验收标准和要求（文档中未提及时注明）
       - 标段划分：标段数量、各标段范围（例如：一标段为3号机组及部分公用系统）
    
    3. 招标范围
       - 具体招标内容：详细列出所有涉及的系统、工程内容（例如：热力、燃料供应、除灰、水处理、供水、电气、热工控制、脱硫、脱硝等系统；物资库房建设；设备代保管；调试工作等）
       - 主要工作量：各项工作的规模或数量（以工程量清单为准，文档中未详细列明时注明）
       - 技术标准：采用的技术规范和标准（文档中未提及时注明）
       - 交付要求：成果交付形式、时间、地点（例如：依据招标文件要求，配合调试等）
       - 调试要求：单体调试、分系统调试、整套启动调试等具体安排
    
    4. 投标人资格要求
       - 资质条件：所需资质等级和类别（例如：电力工程施工总承包一级资质、安全生产许可证、承装一级+承试一级电力设施许可证）
       - 财务要求：财务状况良好（例如：未处于停产、破产状态，资产未被重组、接管等）
       - 业绩要求：近5年类似项目业绩数量和规模（例如：2个及以上1000MW火电机组主体施工已竣工业绩）
       - 项目经理要求：资质、注册建造师、安全生产考核证、无在建项目等（例如：一级注册建造师、安全生产考核B证、近5年1个及以上1000MW机组业绩）
       - 其他要求：是否接受联合体投标、证明材料要求等（例如：不接受联合体；需提供合同、竣工验收证明等）
       - 否决项：可能导致投标被否决的情形（例如：被列入严重失信主体名单、大唐集团“灰名单”“黑名单”）
    
    5. 招标文件获取
       - 获取时间：具体日期和时间范围（例如：2025年5月7日至5月21日17:00）
       - 获取方式：平台网址、需CA证书等（例如：大唐电子商务平台，需企业CA证书下载）
       - 文件售价：费用及发票信息（例如：售价详见平台提示，电子发票发送）
    
    6. 投标文件递交
       - 截止时间：具体日期和时间（例如：2025年5月28日09:00）
       - 递交方式：电子平台、文件大小限制等（例如：大唐电子商务平台，总大小≤800M，单个文件≤100M）
       - 递交地址：平台详情（例如：通过平台电子递交）
       - 保证金：投标保证金金额及缴纳方式（文档中未提及时注明）
       - 文件要求：商务、技术、价格文件的大小、数量、签字盖章要求（例如：总数量≤20个，部分文件需签字盖章）
    
    7. 评标方法和标准
       - 评标方法：如综合评估法（例如：商务10%+技术50%+报价40%）
       - 评分标准：各部分权重、细分项分值（例如：商务部分含企业实力、财务状况；技术部分含施工组织设计；报价部分按偏差打分）
       - 废标条件：具体情形（例如：资格后审未通过、文件不符合要求等）
       - 评标委员会：组成方式（例如：5人以上单数，招标人代表≤1/3，专家随机抽取）
    
    8. 合同条款要点
       - 合同形式：类型和格式（例如：依据招标文件和投标文件签订书面合同）
       - 支付方式：预付款、进度款、竣工结算、质量保证金等详细条件（例如：预付款10%，进度款按月支付85%，竣工结算支付至97%）
       - 履约担保：金额、提交时间、形式（例如：合同金额10%，签订合同前14天内提交）
       - 违约责任：主要违约条款和处罚（例如：逾期支付按中国人民银行基准利率计违约金）
       - 最终结清：缺陷责任期后的支付安排（例如：缺陷责任期12个月，期满后14天内退还保证金）
    
    9. 其他重要信息
       - 踏勘现场：安排（文档中未提及时注明）
       - 答疑安排：时间及方式（文档中未提及时注明）
       - 分包要求：是否允许分包（文档中未提及时注明）
       - 偏差说明：是否允许偏差（文档中未提及时注明）
       - 异议与投诉：提出时间、渠道（例如：对招标文件异议需在投标截止前10日提出；投诉可通过平台或电话）
       - 电子招标投标：平台使用、费用承担（例如：全程电子化，投标人承担投标费用，中标人承担代理服务费）
       - 重新招标与不再招标情形：具体条件（例如：投标人不足3个或所有投标被否决时可重新招标）
    
    要求：
    1. 严格按照上述结构和顺序进行组织
    2. 每个部分都要有具体内容，不能只写标题
    3. 信息要准确，直接来源于招标文件内容
    4. 如某部分内容在招标文件中未提及，请注明“文档中未提及”
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
                st.error("❌ 错误：资源不足！")
                st.info("💡 提示：请补充资源后重试！。")
                raise e
            else:
                st.error(f"发生其他错误: {error_msg}")
                raise e


def main():
    """主函数，运行Streamlit应用程序"""
    st.set_page_config(page_title="招标文件智能分析工具", layout="wide")
    
    st.title("📄 招标文件智能分析工具")
    st.caption("上传招标文件，进行智能分析和总结")
    
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
            # st.success("模型连接成功")
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
