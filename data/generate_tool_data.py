"""Generate 2000+ diverse tool calling training examples for Qwen3 CS model."""
import json, random, statistics

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

# ====== GET_TIME (~150 examples) ======
time_questions = [
    # 基础时间查询
    "现在几点了？", "今天星期几？", "今天是几号？", "现在是什么时间？",
    "告诉我现在的时间", "几点了？", "请问现在几点？", "当前日期和时间是什么？",
    "今天周几啊？", "帮我看看现在的时间", "现在是什么时候了？", "能告诉我现在时间吗？",
    "当前时间是多少？", "现在是几点几分？", "现在是什么日子？", "看一下时间",
    "现在几点几分几秒？", "请问今天是星期几？", "现在几点钟了？",
    # 扩展句式
    "现在北京时间是几点？", "能帮我看看表吗？", "帮我确认一下当前时间",
    "我需要知道现在几点", "告诉我日期", "今天是工作日吗？", "现在是什么季节？",
    "给我报个时", "请问现在日期是？", "今年是哪一年？", "这个月是几月？",
    "明天是星期几？", "昨天是几号？", "三天后是几号？", "下周一是几号？",
    "现在是上午还是下午？", "告诉我今天是几月几号", "现在离过年还有多久？",
    "帮我计时", "现在的时间戳是多少？", "给我当前时间",
    "现在是第几季度？", "这个月有多少天？", "现在是不是快到中午了？",
    "帮忙看下几点了谢谢", "麻烦告诉我现在时间", "我想确认一下现在几点",
    "现在时刻是？", "几点几分了现在？", "现在是什么时辰？",
    "今天农历几号？", "最近是不是快放假了？", "这周是第几周？",
    "告诉我今天的完整日期", "现在的时间精确到秒",
]
extra_time = [
    "能报一下时间吗？", "看看表几点了", "什么时间了现在？", "麻烦看下时间",
    "帮我看看时钟", "目前时间是多少？", "现在是几点钟？", "能告诉我具体时间吗？",
    "看一下手表", "早上还是下午？", "现在是北京时间几点几分？",
    "来报个时间", "此刻时间", "这个点几点了？", "帮看下现在什么时候",
    "现在几点啦？", "帮忙看一下表", "这会几点了？", "当前时刻",
    "告诉我时间吧", "现在这个时间点", "看看几点了现在",
    "现在的时间是几点？", "请问当前时间", "我要对一下时间", "报时",
    "现在几点钟", "现在时间几点", "你那边现在几点？", "系统时间是？",
    "现在是凌晨还是早上？", "到中午了吗？", "下午几点了？", "晚上现在几点？",
    "时间确认一下", "想知道现在时间", "当前几点几分？",
    "给我报个时间吧", "帮忙看下表", "现在大概几点？", "请问现在时刻",
    "麻烦报一下时间", "现在是上午吗？", "几点了看看", "查时间",
    "现在是深夜了吗？", "看下现在几点", "帮我瞅一眼时间",
]
time_questions.extend(extra_time)
random.shuffle(time_questions)

time_obs_responses = [
    ("2026-06-04 08:00:00 Thursday", "现在是2026年6月4日星期四，早上8点整。"),
    ("2026-06-04 08:15:00 Thursday", "现在是早上8点15分，2026年6月4日星期四。"),
    ("2026-06-04 08:45:00 Thursday", "现在是上午8点45分。"),
    ("2026-06-04 09:00:00 Thursday", "现在是上午9点整，星期四。"),
    ("2026-06-04 09:30:00 Thursday", "现在是2026年6月4日上午9点30分。"),
    ("2026-06-04 10:00:00 Thursday", "现在是上午10点整。"),
    ("2026-06-04 10:15:00 Thursday", "现在是上午10点15分，星期四。"),
    ("2026-06-04 11:00:00 Thursday", "现在是上午11点。"),
    ("2026-06-04 11:30:00 Thursday", "现在是上午11点30分。"),
    ("2026-06-04 12:00:00 Thursday", "现在是中午12点整。"),
    ("2026-06-04 13:00:00 Thursday", "现在是下午1点。"),
    ("2026-06-04 14:00:00 Thursday", "现在是下午2点整，2026年6月4日。"),
    ("2026-06-04 14:30:00 Thursday", "现在是下午2点30分。"),
    ("2026-06-04 15:00:00 Thursday", "现在是下午3点整，星期四。"),
    ("2026-06-04 15:45:00 Thursday", "现在是下午3点45分。"),
    ("2026-06-04 16:00:00 Thursday", "现在是下午4点。"),
    ("2026-06-04 16:30:00 Thursday", "现在是下午4点30分，星期四。"),
    ("2026-06-04 17:00:00 Thursday", "现在是下午5点整。"),
    ("2026-06-04 17:45:00 Thursday", "现在是傍晚5点45分。"),
    ("2026-06-04 18:00:00 Thursday", "现在是晚上6点整。"),
    ("2026-06-04 19:00:00 Thursday", "现在是晚上7点。"),
    ("2026-06-04 20:00:00 Thursday", "现在是晚上8点整。"),
    ("2026-06-04 21:30:00 Thursday", "现在是晚上9点30分。"),
    ("2026-06-04 22:00:00 Thursday", "现在是晚上10点整。"),
    ("2026-06-04 23:15:00 Thursday", "现在是晚上11点15分。"),
    ("2026-06-04 Monday", "今天是星期一。"),
    ("2026-06-04 Friday", "今天是星期五。"),
    ("2026-06-04 Saturday", "今天是星期六。"),
    ("2026-06-04 Sunday", "今天是星期日。"),
    ("2026-06-04 Wednesday", "今天是星期三。"),
]

for q in time_questions[:150]:
    obs, resp = random.choice(time_obs_responses)
    add_tool(q, "get_time", {}, obs, resp)

# ====== WEB_SEARCH (~400 examples) ======
search_prefixes = [
    "帮我搜索一下", "查一下", "搜一下", "帮我查查", "搜索",
    "帮我搜一下", "上网查一下", "帮我上网搜", "搜索一下", "查查",
    "用搜索引擎查一下", "在网上找一下", "帮我找找", "查找一下",
    "用百度搜一下", "用谷歌搜索", "帮我查一查",
]

search_topics = [
    ("iPhone 最新评测", "iPhone 评测 2026"),
    ("iPhone 17 Pro 价格", "iPhone 17 Pro 价格 2026"),
    ("华为 Mate 最新款", "华为 Mate 最新款 2026"),
    ("小米汽车 SU7 评测", "小米汽车 SU7 评测"),
    ("MacBook Pro 2026", "MacBook Pro 2026 评测"),
    ("Python 最新版本特性", "Python 最新版本 新特性"),
    ("Python 3.14 发布时间", "Python 3.14 发布 时间"),
    ("Rust 语言最新动态", "Rust 语言 最新 动态 2026"),
    ("Go 语言性能优化", "Go 语言 性能 优化 技巧"),
    ("TypeScript 6.0 新功能", "TypeScript 6.0 新功能"),
    ("今天的科技新闻", "科技新闻 今日"),
    ("全球AI发展最新进展", "AI 发展 最新 进展 2026"),
    ("量子计算突破", "量子计算 最新 突破 2026"),
    ("芯片行业动态", "芯片 行业 动态 2026"),
    ("SpaceX 最新发射", "SpaceX 最新 发射 2026"),
    ("竞争对手最新动态", "电商 行业 动态 2026"),
    ("最新的AI技术发布", "AI 技术 发布 2026"),
    ("DeepSeek 最新版本", "DeepSeek 最新 版本"),
    ("大模型价格对比", "大模型 API 价格 对比 2026"),
    ("开源LLM推荐", "开源 LLM 推荐 2026"),
    ("618促销活动", "618 电商 促销 2026"),
    ("双11活动规则", "双11 活动 规则"),
    ("直播带货技巧", "直播带货 技巧 2026"),
    ("跨境电商新政策", "跨境电商 政策 2026"),
    ("物流行业新政策", "物流 行业 政策 2026"),
    ("最近的天气情况", "北京 天气 近日"),
    ("上海明天天气", "上海 明天 天气"),
    ("深圳台风预警", "深圳 台风 预警 2026"),
    ("成都空气质量", "成都 空气 质量"),
    ("品牌的新闻报道", "品牌 新闻 报道"),
    ("端午节日期", "2026年 端午节 日期"),
    ("中秋节放假安排", "2026年 中秋节 放假 安排"),
    ("国庆节旅游推荐", "国庆节 旅游 推荐 2026"),
    ("春运火车票时间", "2026 春运 火车票 时间"),
    ("法定节假日安排", "2026年 节假日 安排"),
    ("智能手机推荐", "智能手机 推荐 2026"),
    ("智能手表对比", "智能手表 对比 2026"),
    ("无线耳机推荐", "无线 耳机 推荐 2026"),
    ("平板电脑选购指南", "平板电脑 选购 指南 2026"),
    ("扫地机器人评测", "扫地机器人 评测 2026"),
    ("新能源汽车补贴", "新能源汽车 补贴 2026"),
    ("新能源车销量排行", "新能源 车 销量 排行 2026"),
    ("电动车充电桩分布", "充电桩 分布"),
    ("特斯拉新款车型", "特斯拉 新款 车型 2026"),
    ("比亚迪最新技术", "比亚迪 最新 技术"),
    ("最新的游戏大作", "最新游戏 推荐"),
    ("Switch 2 发售日期", "Switch 2 发售 日期"),
    ("Steam 夏季促销", "Steam 夏季 促销 2026"),
    ("PS6 最新消息", "PS6 最新 消息"),
    ("原神最新版本", "原神 最新 版本"),
    ("面试技巧", "求职 面试 技巧"),
    ("简历优化方法", "简历 优化 方法"),
    ("薪资谈判技巧", "薪资 谈判 技巧"),
    ("远程工作机会", "远程 工作 招聘 2026"),
    ("AI行业薪资水平", "AI 行业 薪资 2026"),
    ("理财入门方法", "理财 入门 方法"),
    ("基金定投策略", "基金 定投 策略"),
    ("股市最新行情", "A股 最新 行情 2026"),
    ("数字货币行情", "数字货币 行情 2026"),
    ("怎么买国债", "国债 购买 方法"),
    ("电子书阅读器推荐", "电子书 阅读器 推荐 2026"),
    ("咖啡机品牌排行", "咖啡机 品牌 排行"),
    ("空气炸锅食谱", "空气 炸锅 食谱"),
    ("家用投影仪推荐", "家用 投影仪 推荐 2026"),
    ("智能门锁选购", "智能门锁 选购 指南"),
]

# 每个 topic 用 2-3 个前缀，生成约 400 条
for topic, query in search_topics:
    selected_prefixes = random.sample(search_prefixes, min(6, len(search_prefixes)))
    for prefix in selected_prefixes[:6]:
        q = f"{prefix}{topic}"
        q_variants = [q, f"{q}？", f"请问{q}", f"麻烦{q}", f"能帮我{q}吗？"]
        q = random.choice(q_variants)
        obs_variants = [
            f"搜索结果：已找到关于「{topic}」的相关信息。",
            f"搜索完成，以下是关于{topic}的内容。",
            f"为您找到{topic}相关的多条结果。",
            f"搜索结果返回了{topic}的最新信息。",
        ]
        resp_variants = [
            f"为您搜索到了关于{topic}的信息。",
            f"关于{topic}，搜索到以下内容。",
            f"这是{topic}的搜索结果。",
            f"已为您找到{topic}的相关信息，请查看。",
        ]
        add_tool(random.choice(q_variants), "web_search", {"query": query},
                 random.choice(obs_variants), random.choice(resp_variants))

# ====== SEARCH_KNOWLEDGE_BASE (~400 examples) ======
kb_prefixes = [
    "帮我查一下", "查一下", "查询", "帮我查查", "看看",
    "在知识库中查一下", "检索一下", "查资料", "查文档", "调取",
    "查看内部资料", "调出", "翻阅一下", "查内部系统", "帮我检索",
]

kb_topics = [
    ("退货政策和流程", "退货 政策 流程"),
    ("七天无理由退货条件", "七天 无理由 退货 条件"),
    ("退货运费谁承担", "退货 运费 规则"),
    ("换货流程和时间", "换货 流程 时间"),
    ("物流时效北京到上海", "物流 时效 北京 上海"),
    ("加急配送服务", "加急 配送 服务"),
    ("国际物流时效", "国际 物流 时效"),
    ("大件商品配送", "大件 商品 配送 规则"),
    ("VIP客户权益", "VIP 客户 权益"),
    ("VIP专属客服通道", "VIP 专属 客服 通道"),
    ("会员等级和权益", "会员 等级 权益"),
    ("VIP生日福利", "VIP 生日 福利"),
    ("客服工作时间", "客服 工作 时间"),
    ("夜间客服联系方式", "夜间 客服 联系方式"),
    ("投诉电话是多少", "投诉 电话"),
    ("紧急问题处理通道", "紧急 问题 处理 通道"),
    ("本月促销活动", "当前 促销 活动"),
    ("限时折扣规则", "限时 折扣 规则"),
    ("满减优惠使用说明", "满减 优惠 使用 说明"),
    ("新用户专享优惠", "新用户 专享 优惠"),
    ("退款到账时间", "退款 到账 时间"),
    ("退款方式说明", "退款 方式 说明"),
    ("支付宝退款流程", "支付宝 退款 流程"),
    ("微信支付退款", "微信 支付 退款"),
    ("会员积分规则", "会员 积分 规则"),
    ("积分如何获取", "积分 获取 方式"),
    ("积分有效期", "积分 有效期"),
    ("积分兑换礼品", "积分 兑换 礼品"),
    ("VIP升级条件", "VIP 升级 条件"),
    ("降级规则说明", "VIP 降级 规则"),
    ("会员保级方法", "会员 保级 方法"),
    ("白金卡权益", "白金卡 权益"),
    ("商品库存查询", "商品 库存 查询"),
    ("缺货商品补货时间", "缺货 补货 时间"),
    ("预售商品发货时间", "预售 商品 发货 时间"),
    ("限量商品抢购规则", "限量 商品 抢购 规则"),
    ("配送覆盖城市", "配送 范围 城市"),
    ("偏远地区配送说明", "偏远 地区 配送 说明"),
    ("自提点查询", "自提 点 查询"),
    ("冷链配送范围", "冷链 配送 范围"),
    ("投诉处理流程", "投诉 处理 流程"),
    ("投诉处理时效", "投诉 处理 时效"),
    ("服务质量监督", "服务 质量 监督"),
    ("投诉后如何跟进", "投诉 跟进 方式"),
    ("换货条件和流程", "换货 条件 流程"),
    ("换货包装要求", "换货 包装 要求"),
    ("换货运费规则", "换货 运费 规则"),
    ("跨境商品换货", "跨境 商品 换货"),
    ("售后服务的范围", "售后 服务 范围"),
    ("延保服务说明", "延保 服务 说明"),
    ("维修服务流程", "维修 服务 流程"),
    ("以旧换新政策", "以旧换新 政策"),
    ("大客户优惠政策", "大客户 优惠 政策"),
    ("企业采购折扣", "企业 采购 折扣"),
    ("批量购买优惠", "批量 购买 优惠"),
    ("合同客户权益", "合同 客户 权益"),
    ("修改收货地址方法", "修改 收货 地址 方法"),
    ("订单发货后能否改地址", "发货后 修改 地址"),
    ("合并订单发货", "合并 订单 发货"),
    ("拆分订单配送", "拆分 订单 配送"),
    ("发票开具流程", "发票 开具 流程"),
    ("电子发票下载", "电子 发票 下载"),
    ("增值税发票申请", "增值税 发票 申请"),
    ("发票抬头修改", "修改 发票 抬头"),
    ("账号安全保护", "账号 安全 保护 措施"),
    ("账号被盗怎么办", "账号 被盗 处理"),
    ("修改密码方法", "修改 密码 方法"),
    ("双重认证开启", "双重 认证 开启"),
    ("隐私政策和数据保护", "隐私 政策 数据 保护"),
    ("个人信息删除申请", "个人 信息 删除 申请"),
    ("数据导出方法", "数据 导出 方法"),
    ("Cookie使用说明", "Cookie 使用 说明"),
    ("企业采购流程", "企业 采购 流程 条件"),
    ("企业认证方法", "企业 认证 方法"),
    ("对公转账流程", "对公 转账 流程"),
    ("企业发票专票", "企业 发票 专票"),
    ("礼品卡使用方法", "礼品 卡 使用 方法"),
    ("礼品卡余额查询", "礼品 卡 余额 查询"),
    ("礼品卡过期规则", "礼品 卡 过期 规则"),
    ("礼品卡退换规则", "礼品 卡 退换 规则"),
    ("优惠券使用规则", "优惠券 使用 规则"),
    ("优惠券叠加说明", "优惠券 叠加 说明"),
    ("店铺优惠券和平台券区别", "店铺券 平台券 区别"),
    ("售后工单进度", "售后 工单 进度 查询"),
    ("商品评价规则", "商品 评价 规则"),
    ("晒单返现活动", "晒单 返现 活动"),
    ("以旧换新评估标准", "以旧换新 评估 标准"),
    ("商品保修期限", "商品 保修 期限"),
    ("延保购买方式", "延保 购买 方式"),
]

for topic, query in kb_topics:
    selected_prefixes = random.sample(kb_prefixes, min(5, len(kb_prefixes)))
    for prefix in selected_prefixes[:5]:
        q = f"{prefix}{topic}"
        q_variants = [q, f"{q}？", f"请问{q}", f"麻烦{q}", f"能帮我{q}吗？"]
        q = random.choice(q_variants)
        obs_variants = [
            f"查询结果：关于「{topic}」的详细信息已找到。",
            f"内部知识库返回：已找到{topic}相关文档。",
            f"检索完成，以下是{topic}的具体说明。",
            f"已从知识库中调取{topic}的完整信息。",
        ]
        resp_variants = [
            f"为您查到了关于{topic}的详细信息。",
            f"根据内部资料，{topic}的具体内容如下。",
            f"已为您检索到{topic}的相关文档。",
            f"以下是关于{topic}的说明。",
        ]
        add_tool(random.choice(q_variants), "search_knowledge_base", {"query": query},
                 random.choice(obs_variants), random.choice(resp_variants))

# ====== CODE_INTERPRETER (~500 examples) ======
# NOTE: double braces {{}} in code templates are Python f-string braces escaped for .format()
# Each tuple: (type_id, question_template, code_template, obs_template, [response_templates])
calc_templates = [
    ("discount", "帮我算一下：{price}元的商品打{discount}折后多少钱",
     "price = {price}\ndiscount = {discount}\nfinal = price * discount\nprint(f\"折后价: {{final:.2f}}\")",
     "折后价: {result:.2f}",
     ["折后价格为{result:.2f}元。", "打{discount}折后是{result:.2f}元。"]),
    ("total", "帮我计算：买{count}件商品，单价{unit_price}元，总价多少",
     "count = {count}\nunit_price = {unit_price}\ntotal = count * unit_price\nprint(f\"总价: {{total}}\")",
     "总价: {result}",
     ["总价为{result}元。", "{count}件商品一共{result}元。"]),
    ("installment", "计算一下{price}元分{months}期，年利率{rate}%，每期付多少",
     "P = {price}\nn = {months}\nr = {rate} / 100 / 12\npmt = P * r * (1+r)**n / ((1+r)**n - 1)\nprint(f\"每期: {{pmt:.2f}}\")",
     "每期: {result:.2f}",
     ["每期需要支付{result:.2f}元。", "分{months}期的话，每月还款{result:.2f}元。"]),
    ("mean", "帮我算这组数的平均值：{nums}",
     "import statistics\ndata = [{nums_str}]\nprint(f\"平均值: {{statistics.mean(data):.1f}}\")",
     "平均值: {result:.1f}",
     ["这组数据的平均值是{result:.1f}。", "平均值为{result:.1f}。"]),
    ("power", "计算{base}的{exp}次方",
     "base = {base}\nexp = {exp}\nresult = base ** exp\nprint(f\"结果: {{result}}\")",
     "结果: {result}",
     ["计算结果为{result}。", "{base}的{exp}次方等于{result}。"]),
    ("percent", "{num}的{percent}%是多少",
     "num = {num}\npercent = {percent}\nresult = num * percent / 100\nprint(f\"结果: {{result}}\")",
     "结果: {result}",
     ["{num}的{percent}%是{result}。", "等于{result}。"]),
    ("growth", "从{old_val}增长到{new_val}，增长率是多少？",
     "old = {old_val}\nnew = {new_val}\nrate = (new - old) / old * 100\nprint(f\"增长率: {{rate:.1f}}%\")",
     "增长率: {result:.1f}%",
     ["增长了{result:.1f}%。", "增长率为{result:.1f}%。"]),
    ("compound", "本金{principal}元，年利率{rate2}%，存{years}年，复利后多少钱？",
     "P = {principal}\nr = {rate2} / 100\nn = {years}\nA = P * (1 + r) ** n\nprint(f\"本息合计: {{A:.2f}}\")",
     "本息合计: {result:.2f}",
     ["复利后本息合计{result:.2f}元。", "{years}年后总金额为{result:.2f}元。"]),
    ("stdev", "帮我算这组数据的标准差：{nums}",
     "import statistics\ndata = [{nums_str}]\nprint(f\"标准差: {{statistics.stdev(data):.2f}}\")",
     "标准差: {result:.2f}",
     ["标准差为{result:.2f}。", "数据的标准差是{result:.2f}。"]),
    ("median", "求这组数的中位数：{nums}",
     "import statistics\ndata = [{nums_str}]\nprint(f\"中位数: {{statistics.median(data)}}\")",
     "中位数: {result}",
     ["中位数为{result}。", "这组数据的中位数是{result}。"]),
    ("sort", "帮我把这组数从小到大排序：{nums}",
     "data = [{nums_str}]\nsorted_data = sorted(data)\nprint(f\"排序结果: {{sorted_data}}\")",
     "排序结果: {sorted_str}",
     ["排序后的结果为：{sorted_str}。", "从小到大排列：{sorted_str}。"]),
    ("celsius", "{temp}摄氏度等于多少华氏度？",
     "celsius = {temp}\nfahrenheit = celsius * 9/5 + 32\nprint(f\"华氏度: {{fahrenheit:.1f}}\")",
     "华氏度: {result:.1f}",
     ["{temp}°C = {result:.1f}°F。", "相当于{result:.1f}华氏度。"]),
    ("inch", "{length}英寸是多少厘米？",
     "inches = {length}\ncm = inches * 2.54\nprint(f\"厘米: {{cm:.1f}}\")",
     "厘米: {result:.1f}",
     ["{length}英寸 = {result:.1f}厘米。", "约等于{result:.1f}cm。"]),
    ("bmi", "身高{height}米，体重{weight}公斤，帮我算BMI",
     "height = {height}\nweight = {weight}\nbmi = weight / (height ** 2)\nprint(f\"BMI: {{bmi:.1f}}\")",
     "BMI: {result:.1f}",
     ["您的BMI指数为{result:.1f}。", "BMI = {result:.1f}。"]),
    ("circle", "半径为{radius}的圆面积是多少？",
     "import math\nradius = {radius}\narea = math.pi * radius ** 2\nprint(f\"面积: {{area:.2f}}\")",
     "面积: {result:.2f}",
     ["半径为{radius}的圆面积为{result:.2f}。", "面积约{result:.2f}平方单位。"]),
]

for _ in range(670):
    tid, q_tmpl, code_tmpl, obs_tmpl, resp_opts = random.choice(calc_templates)

    if tid == "discount":
        price = random.randint(50, 20000)
        discount = random.choice([0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95])
        q = q_tmpl.format(price=price, discount=discount)
        code = code_tmpl.format(price=price, discount=discount)
        obs = obs_tmpl.format(result=price * discount)
        resp = random.choice(resp_opts).format(result=price * discount, discount=discount, price=price)
    elif tid == "total":
        count = random.randint(1, 500)
        unit_price = random.randint(5, 10000)
        q = q_tmpl.format(count=count, unit_price=unit_price)
        code = code_tmpl.format(count=count, unit_price=unit_price)
        obs = obs_tmpl.format(result=count * unit_price)
        resp = random.choice(resp_opts).format(result=count * unit_price, count=count, unit_price=unit_price)
    elif tid == "installment":
        price = random.randint(500, 100000)
        months = random.choice([3, 6, 12, 24, 36])
        rate = random.choice([3.6, 4.5, 5.0, 7.2, 9.9, 12.0, 15.0])
        q = q_tmpl.format(price=price, months=months, rate=rate)
        code = code_tmpl.format(price=price, months=months, rate=rate)
        r = rate / 100 / 12
        pmt = price * r * (1+r)**months / ((1+r)**months - 1)
        obs = obs_tmpl.format(result=pmt)
        resp = random.choice(resp_opts).format(result=pmt, months=months)
    elif tid == "mean":
        nums = [random.randint(1, 10000) for _ in range(random.randint(3, 10))]
        nums_str = ", ".join(str(n) for n in nums)
        nums_display = "、".join(str(n) for n in nums)
        q = f"帮我算这组数的平均值：{nums_display}"
        code = code_tmpl.format(nums_str=nums_str)
        obs = obs_tmpl.format(result=statistics.mean(nums))
        resp = random.choice(resp_opts).format(result=statistics.mean(nums))
    elif tid == "power":
        base = random.randint(2, 30)
        exp = random.randint(2, 12)
        q = q_tmpl.format(base=base, exp=exp)
        code = code_tmpl.format(base=base, exp=exp)
        obs = obs_tmpl.format(result=base ** exp)
        resp = random.choice(resp_opts).format(result=base ** exp, base=base, exp=exp)
    elif tid == "percent":
        num = random.randint(100, 100000)
        percent = random.randint(1, 99)
        q = q_tmpl.format(num=num, percent=percent)
        code = code_tmpl.format(num=num, percent=percent)
        obs = obs_tmpl.format(result=num * percent / 100)
        resp = random.choice(resp_opts).format(result=num * percent / 100, num=num, percent=percent)
    elif tid == "growth":
        old_val = random.randint(100, 50000)
        new_val = old_val + random.randint(int(old_val*0.01), int(old_val*0.5))
        q = q_tmpl.format(old_val=old_val, new_val=new_val)
        code = code_tmpl.format(old_val=old_val, new_val=new_val)
        result = (new_val - old_val) / old_val * 100
        obs = obs_tmpl.format(result=result)
        resp = random.choice(resp_opts).format(result=result)
    elif tid == "compound":
        principal = random.randint(1000, 100000)
        rate2 = random.choice([2.0, 2.5, 3.0, 3.5, 4.0, 5.0])
        years = random.choice([1, 2, 3, 5, 10, 20])
        q = q_tmpl.format(principal=principal, rate2=rate2, years=years)
        code = code_tmpl.format(principal=principal, rate2=rate2, years=years)
        result = principal * (1 + rate2/100) ** years
        obs = obs_tmpl.format(result=result)
        resp = random.choice(resp_opts).format(result=result, years=years)
    elif tid == "stdev":
        nums = [random.randint(10, 1000) for _ in range(random.randint(5, 10))]
        nums_str = ", ".join(str(n) for n in nums)
        nums_display = "、".join(str(n) for n in nums)
        q = f"帮我算这组数据的标准差：{nums_display}"
        code = code_tmpl.format(nums_str=nums_str)
        obs = obs_tmpl.format(result=statistics.stdev(nums))
        resp = random.choice(resp_opts).format(result=statistics.stdev(nums))
    elif tid == "median":
        nums = [random.randint(1, 10000) for _ in range(random.randint(4, 10))]
        nums_str = ", ".join(str(n) for n in nums)
        nums_display = "、".join(str(n) for n in nums)
        q = f"求这组数的中位数：{nums_display}"
        code = code_tmpl.format(nums_str=nums_str)
        obs = obs_tmpl.format(result=statistics.median(nums))
        resp = random.choice(resp_opts).format(result=statistics.median(nums))
    elif tid == "sort":
        nums = [random.randint(1, 1000) for _ in range(random.randint(5, 12))]
        nums_str = ", ".join(str(n) for n in nums)
        nums_display = "、".join(str(n) for n in nums)
        q = f"帮我把这组数从小到大排序：{nums_display}"
        code = code_tmpl.format(nums_str=nums_str)
        sorted_nums = sorted(nums)
        sorted_str = ", ".join(str(n) for n in sorted_nums)
        obs = obs_tmpl.format(sorted_str=sorted_str)
        resp = random.choice(resp_opts).format(sorted_str=sorted_str)
    elif tid == "celsius":
        temp = random.randint(-20, 45)
        q = q_tmpl.format(temp=temp)
        code = code_tmpl.format(temp=temp)
        result = temp * 9/5 + 32
        obs = obs_tmpl.format(result=result)
        resp = random.choice(resp_opts).format(result=result, temp=temp)
    elif tid == "inch":
        length = random.randint(1, 100)
        q = q_tmpl.format(length=length)
        code = code_tmpl.format(length=length)
        result = length * 2.54
        obs = obs_tmpl.format(result=result)
        resp = random.choice(resp_opts).format(result=result, length=length)
    elif tid == "bmi":
        height = random.randint(150, 195) / 100
        weight = random.randint(40, 120)
        q = q_tmpl.format(height=height, weight=weight)
        code = code_tmpl.format(height=height, weight=weight)
        result = weight / (height ** 2)
        obs = obs_tmpl.format(result=result)
        resp = random.choice(resp_opts).format(result=result, height=height, weight=weight)
    elif tid == "circle":
        import math
        radius = random.randint(1, 100)
        q = q_tmpl.format(radius=radius)
        code = code_tmpl.format(radius=radius)
        result = math.pi * radius ** 2
        obs = obs_tmpl.format(result=result)
        resp = random.choice(resp_opts).format(result=result, radius=radius)
    add_tool(q, "code_interpreter", {"code": code}, obs, resp)

# ====== DIRECT CHAT (~150 examples) ======
direct_pairs = [
    # 基础问候
    ("你好", "您好！很高兴为您服务，请问有什么可以帮您的吗？"),
    ("你好呀", "您好！欢迎光临，我是您的专属客服助手，有什么可以帮您的？"),
    ("嗨", "嗨！有什么可以帮您的吗？"),
    ("您好", "您好！请问有什么需要？"),
    ("在吗", "在的！随时为您服务。"),
    ("hello", "Hello！有什么可以帮您的？"),
    ("hi", "Hi！很高兴见到您。"),
    # 感谢
    ("谢谢", "不客气！很高兴能帮到您。"),
    ("谢谢你", "不客气！如果还有其他问题，随时找我哦。"),
    ("多谢", "不用谢！这是我应该做的。"),
    ("感谢", "不客气，能帮到您是我的荣幸。"),
    ("太感谢了", "您太客气了！有问题随时找我。"),
    ("十分感谢", "不客气！祝您愉快！"),
    # 告别
    ("好的", "好的，有什么需要随时找我。"),
    ("嗯嗯", "嗯嗯，我在这里随时待命～"),
    ("OK谢谢", "不客气！祝您愉快！"),
    ("好的没别的事了", "好的，那就不打扰您了。再见！"),
    ("拜拜", "再见！祝您生活愉快！"),
    ("再见", "再见！欢迎下次光临。"),
    ("先这样吧", "好的，随时欢迎您再来。"),
    ("那好吧", "好的，祝您愉快！"),
    ("行", "好的！还有什么可以帮您的？"),
    ("好嘞", "好嘞！随时为您服务。"),
    ("嗯好", "嗯好，有需要随时找我。"),
    ("可以了", "好的，还有其他问题吗？"),
    ("暂时没有了", "好的，随时欢迎您再来咨询。"),
    # 夸奖
    ("感谢你的耐心", "这是应该的！欢迎随时再来。"),
    ("你们服务真好", "谢谢您的肯定！我们会继续努力。"),
    ("你回答得真快", "谢谢夸奖！效率是我们的追求。"),
    ("你很专业", "感谢您的认可！我会继续努力。"),
    ("真棒", "谢谢！能帮到您我很开心。"),
    # 犹豫
    ("我再想想", "没问题，您慢慢考虑，有需要随时找我。"),
    ("了解了", "很高兴能帮到您！"),
    ("让我考虑一下", "好的，不着急，我随时都在。"),
    ("我先看看", "没问题，您慢慢浏览，需要帮助随时说。"),
    # 其他
    ("今天心情不好", "听到这个我很抱歉。有什么我能帮您的吗？或者您想聊聊天也可以。"),
    ("能讲个笑话吗", "哈哈，我可能不是最会讲笑话的，但我很乐意帮您解决问题！"),
    ("你叫什么名字", "我是您的AI客服助手，没有特定的名字，您可以叫我小助手！"),
    ("你是真人吗", "我是AI助手，但我能像真人一样帮您解决各种问题哦。"),
    ("你能做什么", "我可以帮您查询订单、解答疑问、计算价格、搜索信息等等，有什么需要尽管说！"),
    ("我需要帮助", "没问题！请告诉我您遇到了什么问题，我会尽力帮您解决。"),
    ("这个产品怎么样", "请问您想了解哪款产品呢？我可以帮您查询详细信息。"),
    ("有没有优惠", "有的！我可以帮您查询当前的优惠活动，请问您对哪类商品感兴趣？"),
    ("发货了吗", "我能理解您关心物流进度！请提供您的订单号，我帮您查询。"),
    ("怎么退货", "退货有专门的流程，我可以帮您查询退货政策和操作步骤，需要吗？"),
    ("投诉在哪里", "非常抱歉给您带来不好的体验。我可以帮您提交投诉，请问是什么问题呢？"),
    # 新增客服互动
    ("我要退货", "好的，请提供您的订单号，我帮您查询退货资格和流程。"),
    ("订单号是20240604001", "好的，已为您查到订单20240604001，请问您需要什么帮助？"),
    ("我要退款", "了解，退款需要走退货流程，商品寄回后我们会尽快处理退款。"),
    ("我的包裹到哪了？", "请提供您的订单号，我帮您查询物流进度。"),
    ("怎么联系人工客服", "您可以直接拨打客服热线，或者我帮您转接人工客服。"),
    ("有什么新品推荐吗？", "请问您对哪类商品感兴趣呢？我帮您搜索一下最新的产品和优惠。"),
    ("周末你们上班吗？", "客服工作时间是每天9:00-22:00，周末也正常服务哦。"),
    ("我要修改订单", "请问您需要修改订单的哪些信息？比如收货地址、商品规格等。"),
    ("怎么领优惠券", "您可以在领券中心领取，或者参加平台的限时活动获取优惠券。"),
    ("帮我看一下这个月花了多少钱", "我可以帮您计算！请提供具体的消费金额明细。"),
    ("商品和描述不符怎么办", "非常抱歉！您可以申请退货退款，我会帮您查询具体的退货流程。"),
    ("团购有优惠吗？", "有的！批量购买可以享受大客户优惠，具体规则我帮您查询一下。"),
    ("帮我看看有没有物流信息", "好的，请提供您的订单号，我帮您查询最新物流状态。"),
    ("怎么取消订单", "您可以在订单详情页申请取消，如果已发货则需要走退货流程。"),
    ("能帮我转人工吗", "当然可以！不过如果不涉及复杂问题，我也可以帮您快速处理哦。"),
    ("我找不到订单", "请不要着急，我帮您查询。您可以通过手机号或用户名查找历史订单。"),
    ("如何成为VIP？", "VIP升级和消费金额、积分相关，我帮您查询具体的升级条件。"),
    ("我的优惠券怎么用不了", "可能是使用条件不满足，我帮您查一下优惠券的具体使用规则。"),
    ("有没有免运费的活动？", "有的！VIP会员享有免邮权益，部分活动商品也有包邮优惠。"),
    ("能帮我催一下快递吗？", "好的，我帮您查询物流情况，如需加急可以联系快递公司处理。"),
    ("换颜色可以吗？", "如果您收货后不满意颜色，可以申请换货，我帮您查询换货流程。"),
    ("你好新年快乐", "新年快乐！祝您新的一年万事如意，有什么可以帮您的？"),
    ("端午节安康", "端午安康！祝您节日愉快，有什么需要随时找我。"),
    ("中秋节快乐", "中秋快乐！祝您阖家团圆，幸福美满。"),
    ("国庆节有什么活动吗", "国庆期间我们有丰富的促销活动，我帮您搜索一下最新的优惠信息。"),
    ("春节放假到什么时候", "春节假期安排已公布，我帮您查一下具体的日期。"),
    ("圣诞节快乐", "圣诞快乐！祝您节日愉快，有什么可以帮您的？"),
    ("情人节有什么推荐吗", "情人节我们有专属礼品推荐和甜蜜优惠，我帮您搜一下热门商品。"),
    ("六一儿童节快乐", "六一快乐！我们有儿童节专题活动，需要帮您看看吗？"),
    ("怎么开发票", "您可以在订单详情页申请电子发票或纸质发票，具体流程我帮您查询。"),
    ("商品收到是坏的", "非常抱歉！请拍照保留证据，我帮您申请换货或退款。"),
    ("怎么修改密码", "您可以在账号设置中修改密码，我帮您查询详细的操作步骤。"),
    ("账号登录不上", "请检查用户名和密码是否正确，如忘记密码可以找回。需要我帮您查询账号安全设置吗？"),
    ("帮我推荐个礼物", "请问是送给谁的呢？我可以根据对象和预算帮您搜索合适的礼物。"),
    ("你们有实体店吗", "有的！我帮您查询离您最近的门店地址。"),
    ("门店地址在哪", "请告诉我您所在的城市，我帮您搜索最近的门店。"),
    ("支持货到付款吗", "部分商品支持货到付款，具体取决于配送地址和商品类型。"),
    ("海外能发货吗", "我们有国际物流服务，具体覆盖国家我帮您查询一下。"),
    ("怎么联系售后", "您可以通过在线客服、电话或邮件联系售后，我也可以直接帮您处理。"),
    ("维修需要多长时间", "维修周期一般为7-15个工作日，具体取决于商品类型和维修内容。"),
    ("换的新品多久到", "换货配送时间与正常订单一致，通常3-5个工作日送达。"),
    ("退款什么时候到账", "退款审核通过后，一般1-7个工作日到账，具体取决于支付方式。"),
    ("能不能便宜点", "价格由商家统一制定，不过我可以帮您查看是否有适用的优惠券或满减活动。"),
    ("好评返现有吗", "部分商品参与晒单返现活动，我帮您查一下当前规则。"),
    ("怎么绑定银行卡", "您可以在支付设置中绑定银行卡，具体步骤我帮您查询。"),
    ("有多少积分", "请提供您的会员账号，我帮您查询当前积分余额。"),
    ("积分快过期了怎么办", "可以赶紧兑换礼品或优惠券哦，我帮您看看有哪些可兑换的。"),
    ("帮我对一下账", "好的，请提供具体的消费明细，我帮您核对。"),
]

for human, assistant in direct_pairs:
    add_direct(human, assistant)

# ====== MULTI-TURN TOOL + CHAT (~250 examples) ======
multi_turn_scenarios = [
    # 模式1: 工具调用 → 追问 → 直接回答（大量）
    ([("帮我查一下会员积分", "search_knowledge_base", {"query": "会员积分"}, "您的积分为3280分", "您的积分为3280分"),
      ("能兑换什么？", "您有3280积分，可兑换30元优惠券、免邮券或小礼品一份。")]),
    ([("搜索一下iPhone 17评测", "web_search", {"query": "iPhone 17 评测"}, "搜索结果：3条相关评测", "找到3条iPhone 17评测"),
      ("第一个评测详细说说", "根据第一条评测，iPhone 17在性能和拍照方面有显著提升。")]),
    ([("现在几点了？", "get_time", {}, "2026-06-04 17:30:00", "现在是下午5点30分"),
      ("好的谢谢", "不客气！祝您愉快！")]),
    ([("帮我算一下199加税8%", "code_interpreter", {"code": "price = 199\ntax = 0.08\ntotal = price * (1 + tax)\nprint(f'含税价: {total:.2f}')"}, "含税价: 214.92", "含税价是214.92元"),
      ("那如果我买3个呢？", "3个的话就是214.92 × 3 = 644.76元。")]),
    # 扩展多轮场景
    ([("查一下我的订单物流", "search_knowledge_base", {"query": "物流 查询 方法"}, "已找到物流查询方法", "您可以通过订单详情页查看物流信息"),
      ("我订单号是20240604001，帮我查", "好的，已为您查询订单20240604001，当前状态：运输中，预计明天到达。")]),
    ([("现在什么时间了？", "get_time", {}, "2026-06-04 09:00:00", "现在是上午9点整"),
      ("那客服上班了吗？", "search_knowledge_base", {"query": "客服工作时间"}, "客服工作时间：每天9:00-22:00", "客服已经上班了，工作时间为每天9:00到22:00，您现在可以联系客服。")]),
    ([("帮我搜一下最新的AI新闻", "web_search", {"query": "AI 新闻 2026"}, "找到5条AI相关新闻", "为您搜索到5条AI最新新闻"),
      ("有关DeepSeek的吗？", "web_search", {"query": "DeepSeek 最新 新闻"}, "找到3条DeepSeek新闻", "有的，找到3条DeepSeek相关新闻。"),
      ("说说第二条", "第二条新闻提到DeepSeek最近发布了新版本，在代码生成方面有重大改进。")]),
    ([("我买的东西坏了怎么办", "search_knowledge_base", {"query": "售后 服务 范围"}, "售后服务涵盖维修、换货、退货", "您可以根据情况选择维修、换货或退货"),
      ("保修期内维修免费吗？", "search_knowledge_base", {"query": "商品 保修 期限"}, "保修期内非人为损坏免费维修", "是的，保修期内非人为损坏可以免费维修。"),
      ("好的，我要申请维修", "好的，请提供您的订单号和商品问题描述，我帮您提交维修申请。")]),
    ([("帮我算下5000元分12期，利率7.2%", "code_interpreter", {"code": "P = 5000\nn = 12\nr = 7.2 / 100 / 12\npmt = P * r * (1+r)**n / ((1+r)**n - 1)\nprint(f'每期: {pmt:.2f}')"}, "每期: 433.67", "每期需要支付433.67元"),
      ("总共要付多少？", "code_interpreter", {"code": "pmt = 433.67\nn = 12\ntotal = pmt * n\nprint(f'总还款: {total:.2f}')"}, "总还款: 5204.04", "分12期总共还款5204.04元，利息合计204.04元。"),
      ("好的了解了", "如果您需要申请分期，可以在结算页面选择分期付款选项。")]),
    # 更多多轮
    ([("帮我查一下退货流程", "search_knowledge_base", {"query": "退货 流程"}, "退货流程：1.申请退货 2.寄回商品 3.验货 4.退款", "退货需要四个步骤：申请退货、寄回商品、验货、退款"),
      ("运费谁出？", "search_knowledge_base", {"query": "退货 运费 规则"}, "质量问题商家承担，非质量问题买家承担", "如果是质量问题，运费由商家承担；非质量问题需要您自行承担哦。")]),
    ([("搜一下618有什么活动", "web_search", {"query": "618 促销 活动 2026"}, "618大促：满300减50，品牌直降", "618活动有满300减50，还有品牌直降优惠"),
      ("我们平台有吗？", "search_knowledge_base", {"query": "当前 促销 活动"}, "平台618活动：跨店满减、限时秒杀、品牌券", "有的！平台618活动包含跨店满减、限时秒杀和品牌优惠券。"),
      ("具体满减规则是什么？", "search_knowledge_base", {"query": "满减 优惠 使用 说明"}, "每满300减50，可跨店累计，上不封顶", "每满300元减50元，可以跨店累计使用，上不封顶哦！")]),
    ([("今天星期几？", "get_time", {}, "2026-06-04 Thursday", "今天是星期四"),
      ("那这周末有什么电影推荐吗？", "web_search", {"query": "本周 电影 推荐 2026"}, "本周热映：《流浪地球3》《封神2》", "这周末热映的有《流浪地球3》和《封神2》，评分都不错！")]),
    ([("帮我看看这个月销售数据平均值：120,340,560,230,780,450", "code_interpreter", {"code": "import statistics\ndata = [120, 340, 560, 230, 780, 450]\nprint(f'平均值: {statistics.mean(data):.1f}')"}, "平均值: 413.3", "这组销售数据的平均值是413.3"),
      ("中位数呢？", "code_interpreter", {"code": "import statistics\ndata = [120, 340, 560, 230, 780, 450]\nprint(f'中位数: {statistics.median(data)}')"}, "中位数: 395.0", "中位数是395.0"),
      ("最高的是哪个月？", "最高值是780，可能对应销售最好的那个月。")]),
]

# Generate more multi-turn scenarios programmatically
for _ in range(350):
    scenario_type = random.randint(1, 10)
    if scenario_type == 1:
        # 时间 + 追问
        obs, resp = random.choice(time_obs_responses)
        follow = random.choice(["好的谢谢", "了解了", "那今天忙不忙？", "还有多久下班？", "那客服还在吗？"])
        follow_resp = random.choice(["不客气！", "祝您愉快！", "还行，有什么需要帮您的？", "客服在工作时间内，随时为您服务。"])
        scenarios = [("现在几点了？", "get_time", {}, obs, resp), (follow, follow_resp)]
    elif scenario_type == 2:
        # 知识库查询 + 追问
        t = random.choice(kb_topics)
        scenarios = [(f"查一下{t[0]}", "search_knowledge_base", {"query": t[1]},
                       f"已找到{t[0]}的详细信息", f"关于{t[0]}的信息已查到"),
                      ("能详细说说吗？", f"为您展开{t[0]}的全部细节内容。")]
    elif scenario_type == 3:
        # 搜索 + 追问
        t = random.choice(search_topics)
        scenarios = [(f"搜一下{t[0]}", "web_search", {"query": t[1]},
                       f"搜索到关于{t[0]}的结果", f"已为您搜索{t[0]}的信息"),
                      ("有什么重点吗？", f"关于{t[0]}，重点内容包括最新动态和热门讨论。")]
    elif scenario_type == 4:
        # 计算 + 追问
        a, b = random.randint(100, 9999), random.randint(100, 9999)
        scenarios = [(f"{a}+{b}等于多少？", "code_interpreter",
                       {"code": f"a = {a}\nb = {b}\nprint(f'{{a+b}}')"},
                       f"{a+b}", f"{a}+{b}={a+b}"),
                      (f"再乘以2呢？", f"{(a+b)*2}。")]
    elif scenario_type == 5:
        # 双工具链：搜索 → 查知识库
        t1 = random.choice(search_topics)
        t2 = random.choice(kb_topics)
        scenarios = [(f"搜一下{t1[0]}", "web_search", {"query": t1[1]},
                       f"搜索结果：{t1[0]}相关资料", f"已搜索到{t1[0]}的外部信息"),
                      (f"内部系统里有{t2[0]}的信息吗？", "search_knowledge_base", {"query": t2[1]},
                       f"已检索到{t2[0]}内部文档", f"内部资料显示{t2[0]}的内容已找到。")]
    elif scenario_type == 6:
        # 时间 + 知识库
        obs1, _ = random.choice(time_obs_responses)
        t2 = random.choice(kb_topics)
        scenarios = [("现在几点了？", "get_time", {}, obs1, f"现在是{obs1}"),
                      (f"查一下{t2[0]}", "search_knowledge_base", {"query": t2[1]},
                       f"已查找到{t2[0]}", f"已为您查到{t2[0]}。")]
    elif scenario_type == 7:
        # 知识库 + 计算
        t1 = random.choice(kb_topics)
        scenarios = [(f"查一下{t1[0]}", "search_knowledge_base", {"query": t1[1]},
                       f"已找到{t1[0]}的信息", f"关于{t1[0]}已查到"),
                      ("帮我算下每月摊下来多少钱", "code_interpreter",
                       {"code": "total = 12000\nmonths = 12\nprint(f'{{total/months:.2f}}')"},
                       "1000.00", "每月约1000元。")]
    elif scenario_type == 8:
        # 知识库 → 知识库（连环查）
        t1 = random.choice(kb_topics)
        t2 = random.choice(kb_topics)
        while t2 == t1:
            t2 = random.choice(kb_topics)
        scenarios = [(f"查一下{t1[0]}", "search_knowledge_base", {"query": t1[1]},
                       f"已查找到{t1[0]}", f"关于{t1[0]}已查到"),
                      (f"那{t2[0]}呢？", "search_knowledge_base", {"query": t2[1]},
                       f"已找到{t2[0]}的文档", f"{t2[0]}的信息也已为您查到。")]
    elif scenario_type == 9:
        # 搜索 → 搜索（连环搜）
        t1 = random.choice(search_topics)
        t2 = random.choice(search_topics)
        while t2 == t1:
            t2 = random.choice(search_topics)
        scenarios = [(f"搜一下{t1[0]}", "web_search", {"query": t1[1]},
                       f"搜索结果已返回", f"已为您搜索{t1[0]}"),
                      (f"再搜一下{t2[0]}", "web_search", {"query": t2[1]},
                       f"搜索完成", f"关于{t2[0]}也已搜索到。")]
    else:
        # 计算 → 计算（连环算）
        a = random.randint(10, 999)
        b = random.randint(10, 999)
        scenarios = [(f"{a}×{b}=？", "code_interpreter",
                       {"code": f"a = {a}\nb = {b}\nprint(f'{{a*b}}')"},
                       f"{a*b}", f"结果是{a*b}"),
                      ("再除以7呢？", "code_interpreter",
                       {"code": f"result = {a*b}\nprint(f'{{result/7:.2f}}')"},
                       f"{a*b/7:.2f}", f"除以7后是{a*b/7:.2f}。")]

    conv = []
    for turn in scenarios:
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
with open("tool_calling_samples.jsonl", "w", encoding="utf-8") as f:
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

multi_count = sum(1 for r in unique if sum(1 for c in r['conversations'] if c['from'] == 'human') > 1)
print(f"  Multi-turn: {multi_count}")
