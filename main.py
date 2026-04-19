# main.py
import asyncio
import json
import time
from openai import OpenAI, AsyncOpenAI
from datasets import load_dataset
import config
import prompts

# ========== 初始化客户端 ==========
client = OpenAI(
    base_url=config.OPENROUTER_BASE_URL,
    api_key=config.OPENROUTER_API_KEY,
)

async_client = AsyncOpenAI(
    base_url=config.OPENROUTER_BASE_URL,
    api_key=config.OPENROUTER_API_KEY,
)


# ========== 工具函数：调用LLM ==========
def call_llm(prompt, system_message="You are a helpful assistant.", temperature=0.3):
    """同步调用LLM"""
    response = client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


async def call_llm_async(prompt, system_message="You are a helpful assistant.", temperature=0.3):
    """异步调用LLM（用于并行）"""
    response = await async_client.chat.completions.create(
        model=config.MODEL_NAME,
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
    )
    return response.choices[0].message.content


def parse_json_response(response_text):
    """从LLM响应中解析JSON"""
    # 尝试提取JSON部分
    text = response_text.strip()
    # 如果响应被```json包裹，去除
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # 如果解析失败，返回原始文本
        print(f"  ⚠️ JSON解析失败，原始响应: {text[:100]}...")
        return {"raw_response": text}


# ========== 步骤1：预处理 ==========
def preprocess_message(raw_message):
    """清洗和规范化用户消息"""
    print("\n📝 [步骤1] 预处理 - 清洗用户消息")
    print(f"  原始消息: {raw_message}")
    
    prompt = prompts.PREPROCESS_PROMPT.format(raw_message=raw_message)
    cleaned = call_llm(prompt, "You are a text preprocessing assistant.")
    
    print(f"  清洗后: {cleaned}")
    return cleaned


# ========== 步骤2：分类 ==========
def classify_message(cleaned_message):
    """分类工单"""
    print("\n🏷️ [步骤2] 分类 - 确定工单类别")
    
    prompt = prompts.CLASSIFICATION_PROMPT.format(cleaned_message=cleaned_message)
    response = call_llm(prompt, "You are a ticket classification assistant.")
    classification = parse_json_response(response)
    
    print(f"  分类结果: {classification}")
    return classification


# ========== 步骤3：路由 + 各分支 ==========
def route_and_generate(message, classification):
    """根据分类路由到不同分支，生成初版回复"""
    category = classification.get("category", "GENERAL")
    branch = config.CATEGORY_TO_BRANCH.get(category, "general")
    
    print(f"\n🔀 [步骤3] 路由 - 类别 '{category}' → 分支 '{branch}'")
    
    # 根据分支选择对应的提示词
    branch_prompts = {
        "technical": prompts.TECHNICAL_PROMPT,
        "billing": prompts.BILLING_PROMPT,
        "general": prompts.GENERAL_PROMPT,
        "complaint": prompts.COMPLAINT_PROMPT,
    }
    
    prompt_template = branch_prompts.get(branch, prompts.GENERAL_PROMPT)
    prompt = prompt_template.format(
        message=message,
        classification=json.dumps(classification, ensure_ascii=False)
    )
    
    system_messages = {
        "technical": "You are a technical support expert.",
        "billing": "You are a billing support specialist.",
        "general": "You are a helpful customer service assistant.",
        "complaint": "You are an empathetic customer service supervisor.",
    }
    
    response = call_llm(prompt, system_messages.get(branch, "You are a helpful assistant."))
    
    print(f"  生成初版回复（分支: {branch}）")
    print(f"  回复预览: {response[:200]}...")
    
    return response, branch


# ========== 步骤4：并行任务 ==========
async def run_parallel_tasks(message):
    """并行执行情感分析和关键词提取"""
    print("\n⚡ [步骤4] 并行处理 - 同时执行情感分析和关键词提取")
    
    start_time = time.time()
    
    # 同时启动两个异步任务
    sentiment_task = call_llm_async(
        prompts.SENTIMENT_PROMPT.format(message=message),
        "You are a sentiment analysis assistant."
    )
    keyword_task = call_llm_async(
        prompts.KEYWORD_PROMPT.format(message=message),
        "You are a keyword extraction assistant."
    )
    
    # 等待两个任务都完成
    sentiment_response, keyword_response = await asyncio.gather(
        sentiment_task, keyword_task
    )
    
    elapsed = time.time() - start_time
    
    sentiment = parse_json_response(sentiment_response)
    keywords = parse_json_response(keyword_response)
    
    print(f"  ✅ 并行任务完成 (耗时 {elapsed:.2f}秒)")
    print(f"  情感分析: {sentiment}")
    print(f"  关键词: {keywords}")
    
    return sentiment, keywords


# ========== 步骤5：反思循环 ==========
def reflection_loop(message, classification, initial_response, max_iterations=2):
    """
    反思循环：评估 → 改进 → 再评估
    至少迭代2次
    """
    print("\n🔄 [步骤5] 反思循环 - 自我评估与改进")
    
    current_response = initial_response
    
    for iteration in range(1, max_iterations + 1):
        print(f"\n  --- 迭代 {iteration} ---")
        
        # 评估当前回复
        print(f"  📊 评估中...")
        eval_prompt = prompts.EVALUATION_PROMPT.format(
            message=message,
            classification=json.dumps(classification, ensure_ascii=False),
            response=current_response
        )
        eval_response = call_llm(eval_prompt, "You are a quality evaluation expert.")
        evaluation = parse_json_response(eval_response)
        
        print(f"  评分: 语气={evaluation.get('tone_score', 'N/A')}/10, "
              f"完整性={evaluation.get('completeness_score', 'N/A')}/10, "
              f"准确性={evaluation.get('accuracy_score', 'N/A')}/10")
        print(f"  总分: {evaluation.get('total_score', 'N/A')}")
        print(f"  评价: {evaluation.get('critique', '无')[:200]}...")
        
        # 判断是否需要改进
        if not evaluation.get("needs_improvement", True):
            print(f"  ✅ 回复已达标，停止迭代")
            break
        
        # 生成改进后的回复
        if iteration < max_iterations:
            print(f"  🔧 改进中...")
            improve_prompt = prompts.IMPROVEMENT_PROMPT.format(
                response=current_response,
                critique=evaluation.get("critique", "需要改进"),
                message=message,
                classification=json.dumps(classification, ensure_ascii=False)
            )
            improved_response = call_llm(improve_prompt, "You are a response improvement expert.")
            
            # 显示改进前后的对比
            print(f"  📝 改进前: {current_response[:150]}...")
            print(f"  📝 改进后: {improved_response[:150]}...")
            
            current_response = improved_response
    
    return current_response, evaluation


# ========== 主流程 ==========
async def process_ticket(raw_message):
    """处理单个工单的完整流程"""
    print("\n" + "="*60)
    print(f"🚀 开始处理工单")
    print("="*60)
    
    # Step 1: 预处理
    cleaned = preprocess_message(raw_message)
    
    # Step 2: 分类
    classification = classify_message(cleaned)
    
    # Step 3: 路由 + 生成初版回复
    draft_response, branch = route_and_generate(cleaned, classification)
    
    # Step 4: 并行任务
    sentiment, keywords = await run_parallel_tasks(cleaned)
    
    # Step 5: 反思循环
    final_response, evaluation = reflection_loop(cleaned, classification, draft_response)
    
    # 汇总结果
    result = {
        "original_message": raw_message,
        "cleaned_message": cleaned,
        "classification": classification,
        "branch": branch,
        "sentiment": sentiment,
        "keywords": keywords,
        "draft_response": draft_response,
        "final_response": final_response,
        "evaluation": evaluation,
    }
    
    print("\n" + "="*60)
    print("✅ 工单处理完成！")
    print("="*60)
    print(f"\n📋 最终回复:\n{final_response}\n")
    
    return result


# ========== 测试数据（至少10条，覆盖所有分支）==========
SAMPLE_TICKETS = [
    # 技术问题 (TECHNICAL)
    "I can't log into my account. The app keeps saying 'invalid password' even though I'm sure it's correct. Pls help!",
    "Your website is so slow today. I've been trying to checkout for 20 minutes!",
    "The mobile app crashes every time I try to open it. What's going on?",
    
    # 账单/退款 (BILLING)
    "I was charged twice for my subscription this month. Can you refund the extra charge?",
    "How do I cancel my subscription and get a refund? I don't use the service anymore.",
    "My invoice shows the wrong amount. It should be $29.99 but you charged me $49.99.",
    
    # 一般咨询 (GENERAL)
    "What are your business hours? I need to know when I can call support.",
    "Do you ship internationally? I'm in Canada and want to order something.",
    "What's the difference between the basic and premium plans?",
    
    # 投诉 (COMPLAINT)
    "This is ridiculous! I've been waiting for a response for 3 days and nobody has helped me!",
    "Your customer service is terrible. I want to speak to a manager immediately!",
    "I'm extremely disappointed with the quality of this product. It broke after one week of use.",
]


# ========== 主函数 ==========
async def main():
    """运行演示"""
    print("\n🎯 客服工单处理系统演示")
    print(f"📦 使用模型: {config.MODEL_NAME}")
    print(f"📋 测试工单数量: {len(SAMPLE_TICKETS)}")
    
    # 处理前3个工单作为演示（可以改成处理全部）
    demo_tickets = SAMPLE_TICKETS[:3]  # 先演示3个
    
    results = []
    for i, ticket in enumerate(demo_tickets, 1):
        print(f"\n\n{'='*60}")
        print(f"📨 处理第 {i}/{len(demo_tickets)} 个工单")
        print(f"{'='*60}")
        
        result = await process_ticket(ticket)
        results.append(result)
        
        # 工单之间稍作停顿
        await asyncio.sleep(1)
    
    print("\n\n" + "="*60)
    print("🎉 所有演示工单处理完成！")
    print("="*60)
    
    return results


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())