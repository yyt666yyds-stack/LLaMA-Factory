"""Generate 800+ diverse tool calling training examples for Qwen3 CS model."""
import json, random, re

random.seed(42)

records = []

def add_tool(human, tool_name, tool_args, observation, assistant):
    records.append({"conversations": [
        {"from": "human", "value": human},
        {"from": "function_call", "value": json.dumps([{"name": tool_name, "arguments": tool_args}], ensure_ascii=False)},
        {"from": "observation", "value": observation},
        {"from": "gpt", "value": assistant},
    ], "system": ""})

def add_direct(human, assistant):
    records.append({"conversations": [
        {"from": "human", "value": human},
        {"from": "gpt", "value": assistant},
    ], "system": ""})

# ====== GET_TIME (100 examples) ======
time_questions = [
    "现在几点了？", "今天星期几？", "今天是几号？", "现在是什么时间？",
    "告诉我现在的时间", "几点了？", "请问现在几点？", "当前日期和时间是什么？",
    "今天周几啊？", "帮我看看现在的时间", "现在是什么时候了？", "能告诉我现在时间吗？",
    "我要记录一下时间", "当前时间是多少？", "现在是几点几分？", "现在是什么日子？",
    "看一下时间", "现在几点几分几秒？", "请问今天是星期几？", "现在几点钟了？",
]
time_obs = [
    ("2026-06-04 08:00:00", "早上8点整"),
    ("2026-06-04 10:30:00", "上午10点30分"),
    ("2026-06-04 14:00:00", "下午2点整"),
    ("2026-06-04 17:00:00", "下午5点整"),
    ("2026-06-04 20:00:00", "晚上8点整"),
    ("2026-06-04 Monday", "星期一"),
    ("2026-06-04 Friday", "星期五"),
    ("2026-06-04 09:15:00", "上午9点15分"),
    ("2026-06-04 12:00:00", "中午12点"),
    ("2026-06-04 22:30:00", "晚上10点30分"),
]
for q in time_questions:
    obs, resp = random.choice(time_obs)
    add_tool(q, "get_time", {}, obs, f"现在是{resp}。")

# ====== WEB_SEARCH (200 examples) ======
search_topics = [
    ("iPhone 最新评测", "iPhone 最新评测 2026"),
    ("Python 最新版本特性", "Python 最新版本 新特性"),
    ("今天的科技新闻", "科技新闻 今日"),
    ("竞争对手最新动态", "电商平台 行业动态 2026"),
    ("最新的AI技术发布", "AI 技术 发布 2026"),
    ("618促销活动", "618 电商 促销 2026"),
    ("最近的天气情况", "北京 天气 近日"),
    ("品牌的新闻报道", "电商平台 品牌 新闻"),
    ("端午节日期", "2026年 端午节 日期"),
    ("物流行业新政策", "物流 行业 政策 2026"),
    ("智能手机推荐", "智能手机 推荐 2026"),
    ("直播带货趋势", "直播带货 趋势 2026"),
    ("新能源汽车补贴", "新能源汽车 补贴 2026"),
    ("法定节假日安排", "2026年 节假日 安排"),
    ("最新的游戏大作", "最新游戏 推荐"),
    ("外卖优惠券", "外卖 优惠券"),
    ("面试技巧", "求职 面试 技巧"),
    ("理财入门方法", "理财 入门 方法"),
    ("电子书阅读器推荐", "电子书 阅读器 推荐 2026"),
    ("咖啡机品牌排行", "咖啡机 品牌 排行"),
]
for topic, query in search_topics:
    for prefix in ["帮我搜索一下", "查一下", "搜一下", "帮我查查", "搜索"]:
        q = f"{prefix}{topic}"
        obs = f"搜索结果：已找到关于「{topic}」的相关信息。"
        resp = f"为您搜索到了关于{topic}的信息，请查看。"
        add_tool(q, "web_search", {"query": query}, obs, resp)

# ====== SEARCH_KNOWLEDGE_BASE (200 examples) ======
kb_topics = [
    ("退货政策和流程", "退货政策 流程"),
    ("物流时效北京到上海", "物流时效 北京 上海"),
    ("VIP客户权益", "VIP客户权益"),
    ("客服工作时间", "客服工作时间"),
    ("本月促销活动", "当前促销活动"),
    ("退款到账时间", "退款 到账时间"),
    ("会员积分规则", "会员积分 规则"),
    ("VIP升级条件", "VIP会员 升级条件"),
    ("商品库存查询", "商品 库存 查询"),
    ("配送覆盖城市", "配送范围 城市"),
    ("投诉处理流程", "投诉处理 流程"),
    ("换货条件和流程", "换货 条件 流程"),
    ("售后服务的范围", "售后服务 范围"),
    ("大客户优惠政策", "大客户 优惠政策"),
    ("修改收货地址方法", "修改 收货地址 方法"),
    ("发票开具流程", "发票 开具 流程"),
    ("账号安全保护", "账号安全 保护措施"),
    ("隐私政策和数据保护", "隐私政策 数据保护"),
    ("企业采购流程", "企业采购 流程 条件"),
    ("礼品卡使用方法", "礼品卡 使用 方法"),
]
for topic, query in kb_topics:
    for prefix in ["帮我查一下", "查一下", "查询", "帮我查查", "看看"]:
        q = f"{prefix}{topic}"
        obs = f"查询结果：关于「{topic}」的详细信息已找到。"
        resp = f"为您查到了关于{topic}的详细信息。"
        add_tool(q, "search_knowledge_base", {"query": query}, obs, resp)

# ====== CODE_INTERPRETER (200 examples) ======
calc_templates = [
    ("帮我算一下：{price}元的商品打{discount}折后多少钱",
     'price = {price}\ndiscount = {discount}\nfinal = price * discount\nprint(f"折后价: {final:.2f}")',
     "折后价: {result:.2f}"),
    ("帮我计算：买{count}件商品，单价{unit_price}元，总价多少",
     'count = {count}\nunit_price = {unit_price}\ntotal = count * unit_price\nprint(f"总价: {total}")',
     "总价: {result}"),
    ("计算一下{price}元分{months}期，年利率{rate}%，每期付多少",
     'P = {price}\nn = {months}\nr = {rate} / 100 / 12\npmt = P * r * (1+r)**n / ((1+r)**n - 1)\nprint(f"每期: {pmt:.2f}")',
     "每期: {result:.2f}"),
    ("帮我算这组数的平均值：{nums}",
     'import statistics\ndata = [{nums_str}]\nprint(f"平均值: {statistics.mean(data):.1f}")',
     "平均值: {result:.1f}"),
    ("计算{base}的{exp}次方",
     'base = {base}\nexp = {exp}\nresult = base ** exp\nprint(f"结果: {result}")',
     "结果: {result}"),
]
for _ in range(200):
    t = random.choice(calc_templates)
    if "折后价" in t[0]:
        price = random.randint(100, 10000)
        discount = random.choice([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
        q = t[0].format(price=price, discount=discount)
        code = t[1].format(price=price, discount=discount)
        result = price * discount
        obs = t[2].format(result=result)
        resp = f"折后价格为{result:.2f}元。"
    elif "总价" in t[0]:
        count = random.randint(1, 200)
        unit_price = random.randint(10, 5000)
        q = t[0].format(count=count, unit_price=unit_price)
        code = t[1].format(count=count, unit_price=unit_price)
        result = count * unit_price
        obs = t[2].format(result=result)
        resp = f"总价为{result}元。"
    elif "分期" in t[0]:
        price = random.randint(1000, 50000)
        months = random.choice([3, 6, 12, 24, 36])
        rate = random.choice([3.6, 5.0, 7.2, 9.9, 12.0])
        q = t[0].format(price=price, months=months, rate=rate)
        code = t[1].format(price=price, months=months, rate=rate)
        r = rate / 100 / 12
        pmt = price * r * (1+r)**months / ((1+r)**months - 1)
        obs = t[2].format(result=pmt)
        resp = f"每期需要支付{pmt:.2f}元。"
    elif "平均值" in t[0]:
        nums = [random.randint(1, 1000) for _ in range(random.randint(3, 8))]
        nums_str = ", ".join(str(n) for n in nums)
        nums_display = "、".join(str(n) for n in nums)
        q = f"帮我算这组数的平均值：{nums_display}"
        code = t[1].format(nums_str=nums_str)
        import statistics
        result = statistics.mean(nums)
        obs = t[2].format(result=result)
        resp = f"这组数据的平均值是{result:.1f}。"
    else:
        base = random.randint(2, 20)
        exp = random.randint(2, 10)
        q = t[0].format(base=base, exp=exp)
        code = t[1].format(base=base, exp=exp)
        result = base ** exp
        obs = t[2].format(result=result)
        resp = f"计算结果为{result}。"
    add_tool(q, "code_interpreter", {"code": code}, obs, resp)

# ====== DIRECT CHAT (100 examples) ======
direct_pairs = [
    ("你好", "您好！很高兴为您服务，请问有什么可以帮您的吗？"),
    ("你好呀", "您好！欢迎光临，我是您的专属客服助手，有什么可以帮您的？"),
    ("谢谢", "不客气！很高兴能帮到您。"),
    ("谢谢你", "不客气！如果还有其他问题，随时找我哦。"),
    ("好的", "好的，有什么需要随时找我。"),
    ("嗯嗯", "嗯嗯，我在这里随时待命～"),
    ("OK谢谢", "不客气！祝您愉快！"),
    ("好的没别的事了", "好的，那就不打扰您了。再见！"),
    ("拜拜", "再见！祝您生活愉快！"),
    ("感谢你的耐心", "这是应该的！欢迎随时再来。"),
    ("你们服务真好", "谢谢您的肯定！我们会继续努力。"),
    ("我再想想", "没问题，您慢慢考虑，有需要随时找我。"),
    ("先这样吧", "好的，随时欢迎您再来。"),
    ("了解了", "很高兴能帮到您！"),
    ("行", "好的！还有什么可以帮您的？"),
    ("好嘞", "好嘞！随时为您服务。"),
    ("嗯好", "嗯好，有需要随时找我。"),
    ("那好吧", "好的，祝您愉快！"),
    ("可以了", "好的，还有其他问题吗？"),
    ("暂时没有了", "好的，随时欢迎您再来咨询。"),
]
for human, assistant in direct_pairs:
    add_direct(human, assistant)

# ====== MULTI-TURN TOOL + CHAT (50 examples) ======
multi_turn_scenarios = [
    ([("帮我查一下会员积分", "search_knowledge_base", {"query": "会员积分"}, "您的积分为3280分", "您的积分为3280分"),
      ("能兑换什么？", "您有3280积分，可兑换30元优惠券或免邮券。")]),
    ([("搜索一下iPhone 17评测", "web_search", {"query": "iPhone 17 评测"}, "搜索结果：3条相关评测", "找到3条iPhone 17评测"),
      ("第一个评测详细说说", "根据第一条评测，iPhone 17在性能和拍照方面有显著提升。")]),
    ([("现在几点了？", "get_time", {}, "2026-06-04 17:30:00", "现在是下午5点30分"),
      ("好的谢谢", "不客气！祝您愉快！")]),
    ([("帮我算一下199加税8%", "code_interpreter", {"code": "price = 199\ntax = 0.08\ntotal = price * (1 + tax)\nprint(f'含税价: {total:.2f}')"}, "含税价: 214.92", "含税价是214.92元"),
      ("那如果我买3个呢？", "3个的话就是214.92 × 3 = 644.76元。")]),
]
for turns in multi_turn_scenarios:
    conv = []
    for turn in turns:
        if len(turn) == 2:
            conv.append({"from": "human", "value": turn[0]})
            conv.append({"from": "gpt", "value": turn[1]})
        else:
            human, tool_name, tool_args, obs, resp = turn
            conv.append({"from": "human", "value": human})
            conv.append({"from": "function_call", "value": json.dumps([{"name": tool_name, "arguments": tool_args}], ensure_ascii=False)})
            conv.append({"from": "observation", "value": obs})
            conv.append({"from": "gpt", "value": resp})
    records.append({"conversations": conv, "system": ""})

# Deduplicate
seen = set()
unique = []
for r in records:
    key = json.dumps(r["conversations"], ensure_ascii=False, sort_keys=True)
    if key not in seen:
        seen.add(key)
        unique.append(r)

# Save
with open("/mnt/workspace/LLaMA-Factory/data/tool_calling_samples.jsonl", "w") as f:
    for r in unique:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

tool_count = sum(1 for r in unique if any(c['from'] == 'function_call' for c in r['conversations']))
direct_count = len(unique) - tool_count
print(f"Total examples: {len(unique)}")
print(f"  Tool-calling: {tool_count}")
print(f"  Direct chat: {direct_count}")

tc_dist = {}
for r in unique:
    for c in r["conversations"]:
        if c["from"] == "function_call":
            data = json.loads(c["value"])
            for tc in (data if isinstance(data, list) else [data]):
                tc_dist[tc["name"]] = tc_dist.get(tc["name"], 0) + 1
print(f"  Tool distribution: {tc_dist}")
