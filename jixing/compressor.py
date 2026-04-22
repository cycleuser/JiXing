"""Semantic text compression for context window management.

Compresses text by expressing the same meaning with fewer tokens,
using shorter expressions in the same language while preserving meaning.

Supports Chinese and English text compression.
"""

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class CompressionRule:
    """A rule for text compression."""

    pattern: str
    replacement: str
    description: str
    language: str = "zh"
    priority: int = 1


@dataclass
class CompressionResult:
    """Result of text compression."""

    original_text: str
    compressed_text: str
    original_length: int
    compressed_length: int
    compression_ratio: float
    rules_applied: list[str]
    token_reduction: int
    meaning_preserved: bool = True


ZH_COMPRESSION_RULES = [
    CompressionRule(
        pattern=r"首先，",
        replacement="",
        description="删除机械过渡词'首先'",
        language="zh",
        priority=1,
    ),
    CompressionRule(
        pattern=r"其次，",
        replacement="另外，",
        description="替换'其次'为更自然的'另外'",
        language="zh",
        priority=1,
    ),
    CompressionRule(
        pattern=r"最后，",
        replacement="最后",
        description="删除'最后'后的逗号",
        language="zh",
        priority=1,
    ),
    CompressionRule(
        pattern=r"综上所述，?",
        replacement="总之",
        description="替换'综上所述'为更简短的'总之'",
        language="zh",
        priority=1,
    ),
    CompressionRule(
        pattern=r"总而言之，?",
        replacement="总之",
        description="替换'总而言之'为更简短的'总之'",
        language="zh",
        priority=1,
    ),
    CompressionRule(
        pattern=r"一方面，?([^，]+)；?另一方面，?",
        replacement=r"\1，同时",
        description="压缩'一方面...另一方面'结构",
        language="zh",
        priority=2,
    ),
    CompressionRule(
        pattern=r"具有(.{1,10})的特点",
        replacement=r"\1",
        description="删除'具有...的特点'冗余结构",
        language="zh",
        priority=2,
    ),
    CompressionRule(
        pattern=r"进行(.{1,10})的处理",
        replacement=r"处理\1",
        description="简化'进行...的处理'为'处理...'",
        language="zh",
        priority=2,
    ),
    CompressionRule(
        pattern=r"对(.{1,10})进行了(.{1,10})",
        replacement=r"\2了\1",
        description="简化'对...进行了...'结构",
        language="zh",
        priority=2,
    ),
    CompressionRule(
        pattern=r"从(.{1,10})角度来看",
        replacement=r"看\1",
        description="简化'从...角度来看'",
        language="zh",
        priority=2,
    ),
    CompressionRule(
        pattern=r"在(.{1,10})方面",
        replacement=r"\1上",
        description="简化'在...方面'为'...上'",
        language="zh",
        priority=2,
    ),
    CompressionRule(
        pattern=r"对于(.{1,10})来说",
        replacement=r"对\1",
        description="简化'对于...来说'为'对...'",
        language="zh",
        priority=2,
    ),
    CompressionRule(
        pattern=r"是一个(.{1,10})的过程",
        replacement=r"需要\1",
        description="简化'是一个...的过程'",
        language="zh",
        priority=2,
    ),
    CompressionRule(
        pattern=r"有着(.{1,10})的作用",
        replacement=r"能\1",
        description="简化'有着...的作用'",
        language="zh",
        priority=2,
    ),
    CompressionRule(
        pattern=r"的实现",
        replacement="实现",
        description="删除冗余的'的'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的情况下",
        replacement="时",
        description="简化'的情况下'为'时'",
        language="zh",
        priority=2,
    ),
    CompressionRule(
        pattern=r"的基础上",
        replacement="上",
        description="简化'的基础上'为'上'",
        language="zh",
        priority=2,
    ),
    CompressionRule(
        pattern=r"的过程中",
        replacement="时",
        description="简化'的过程中'为'时'",
        language="zh",
        priority=2,
    ),
    CompressionRule(
        pattern=r"的可能性",
        replacement="可能",
        description="简化'的可能性'为'可能'",
        language="zh",
        priority=2,
    ),
    CompressionRule(
        pattern=r"的重要性",
        replacement="重要",
        description="简化'的重要性'为'重要'",
        language="zh",
        priority=2,
    ),
    CompressionRule(
        pattern=r"的必要性",
        replacement="必须",
        description="简化'的必要性'为'必须'",
        language="zh",
        priority=2,
    ),
    CompressionRule(
        pattern=r"的有效方法",
        replacement="好方法",
        description="简化'的有效方法'为'好方法'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的重要手段",
        replacement="好手段",
        description="简化'的重要手段'为'好手段'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的关键因素",
        replacement="关键",
        description="简化'的关键因素'为'关键'",
        language="zh",
        priority=2,
    ),
    CompressionRule(
        pattern=r"的核心技术",
        replacement="核心技术",
        description="保留核心技术",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的发展趋势",
        replacement="趋势",
        description="简化'的发展趋势'为'趋势'",
        language="zh",
        priority=2,
    ),
    CompressionRule(
        pattern=r"的应用场景",
        replacement="场景",
        description="简化'的应用场景'为'场景'",
        language="zh",
        priority=2,
    ),
    CompressionRule(
        pattern=r"的实际应用",
        replacement="应用",
        description="简化'的实际应用'为'应用'",
        language="zh",
        priority=2,
    ),
    CompressionRule(
        pattern=r"的具体实现",
        replacement="实现",
        description="简化'的具体实现'为'实现'",
        language="zh",
        priority=2,
    ),
    CompressionRule(
        pattern=r"的详细说明",
        replacement="说明",
        description="简化'的详细说明'为'说明'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的深入分析",
        replacement="分析",
        description="简化'的深入分析'为'分析'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的全面介绍",
        replacement="介绍",
        description="简化'的全面介绍'为'介绍'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的广泛使用",
        replacement="常用",
        description="简化'的广泛使用'为'常用'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的快速发展",
        replacement="快发展",
        description="简化'的快速发展'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的巨大变化",
        replacement="大变",
        description="简化'的巨大变化'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的显著提升",
        replacement="提升",
        description="简化'的显著提升'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的明显改善",
        replacement="改善",
        description="简化'的明显改善'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的有效解决",
        replacement="解决",
        description="简化'的有效解决'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的广泛应用",
        replacement="常用",
        description="简化'的广泛应用'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的深入研究",
        replacement="研究",
        description="简化'的深入研究'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的广泛讨论",
        replacement="讨论",
        description="简化'的广泛讨论'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的高度关注",
        replacement="关注",
        description="简化'的高度关注'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的广泛关注",
        replacement="关注",
        description="简化'的广泛关注'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的普遍认可",
        replacement="认可",
        description="简化'的普遍认可'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的一致认同",
        replacement="认同",
        description="简化'的一致认同'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的充分肯定",
        replacement="肯定",
        description="简化'的充分肯定'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的高度重视",
        replacement="重视",
        description="简化'的高度重视'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的大力支持",
        replacement="支持",
        description="简化'的大力支持'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的积极推动",
        replacement="推动",
        description="简化'的积极推动'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的促进",
        replacement="促",
        description="简化'的促进'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的发展",
        replacement="发展",
        description="保留发展",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的实现",
        replacement="实现",
        description="保留实现",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的应用",
        replacement="应用",
        description="保留应用",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的研究",
        replacement="研究",
        description="保留研究",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的探讨",
        replacement="探讨",
        description="保留探讨",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的分析",
        replacement="分析",
        description="保留分析",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的介绍",
        replacement="介绍",
        description="保留介绍",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的说明",
        replacement="说明",
        description="保留说明",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的描述",
        replacement="描述",
        description="保留描述",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的阐述",
        replacement="阐述",
        description="保留阐述",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的论述",
        replacement="论述",
        description="保留论述",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的讨论",
        replacement="讨论",
        description="保留讨论",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的交流",
        replacement="交流",
        description="保留交流",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的合作",
        replacement="合作",
        description="保留合作",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的协作",
        replacement="协作",
        description="保留协作",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的配合",
        replacement="配合",
        description="保留配合",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的协调",
        replacement="协调",
        description="保留协调",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的整合",
        replacement="整合",
        description="保留整合",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的融合",
        replacement="融合",
        description="保留融合",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的结合",
        replacement="结合",
        description="保留结合",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的统一",
        replacement="统一",
        description="保留统一",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的整合",
        replacement="整合",
        description="保留整合",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的优化",
        replacement="优化",
        description="保留优化",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的改进",
        replacement="改进",
        description="保留改进",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的完善",
        replacement="完善",
        description="保留完善",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的提升",
        replacement="提升",
        description="保留提升",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的提高",
        replacement="提高",
        description="保留提高",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的增强",
        replacement="增强",
        description="保留增强",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的加强",
        replacement="加强",
        description="保留加强",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的强化",
        replacement="强化",
        description="保留强化",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的深化",
        replacement="深化",
        description="保留深化",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的细化",
        replacement="细化",
        description="保留细化",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的具体化",
        replacement="具体",
        description="简化'的具体化'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的明确化",
        replacement="明确",
        description="简化'的明确化'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的规范化",
        replacement="规范",
        description="简化'的规范化'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的标准化",
        replacement="标准",
        description="简化'的标准化'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的制度化",
        replacement="制度",
        description="简化'的制度化'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的程序化",
        replacement="程序",
        description="简化'的程序化'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的自动化",
        replacement="自动",
        description="简化'的自动化'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的智能化",
        replacement="智能",
        description="简化'的智能化'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的数字化",
        replacement="数字",
        description="简化'的数字化'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的网络化",
        replacement="网络",
        description="简化'的网络化'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的信息化",
        replacement="信息",
        description="简化'的信息化'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的电子化",
        replacement="电子",
        description="简化'的电子化'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的现代化",
        replacement="现代",
        description="简化'的现代化'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的科技化",
        replacement="科技",
        description="简化'的科技化'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的科学化",
        replacement="科学",
        description="简化'的科学化'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的合理化",
        replacement="合理",
        description="简化'的合理化'",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的优化",
        replacement="优化",
        description="保留优化",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的精简",
        replacement="精简",
        description="保留精简",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的简化",
        replacement="简化",
        description="保留简化",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的压缩",
        replacement="压缩",
        description="保留压缩",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的缩减",
        replacement="缩减",
        description="保留缩减",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的缩小",
        replacement="缩小",
        description="保留缩小",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的减少",
        replacement="减少",
        description="保留减少",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的降低",
        replacement="降低",
        description="保留降低",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的下降",
        replacement="下降",
        description="保留下降",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的减少",
        replacement="减少",
        description="保留减少",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的消除",
        replacement="消除",
        description="保留消除",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的避免",
        replacement="避免",
        description="保留避免",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的防止",
        replacement="防止",
        description="保留防止",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的预防",
        replacement="预防",
        description="保留预防",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的控制",
        replacement="控制",
        description="保留控制",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的管理",
        replacement="管理",
        description="保留管理",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的监督",
        replacement="监督",
        description="保留监督",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的检查",
        replacement="检查",
        description="保留检查",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的审查",
        replacement="审查",
        description="保留审查",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的审核",
        replacement="审核",
        description="保留审核",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的审批",
        replacement="审批",
        description="保留审批",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的批准",
        replacement="批准",
        description="保留批准",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的认可",
        replacement="认可",
        description="保留认可",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的确认",
        replacement="确认",
        description="保留确认",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的验证",
        replacement="验证",
        description="保留验证",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的检验",
        replacement="检验",
        description="保留检验",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的测试",
        replacement="测试",
        description="保留测试",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的试验",
        replacement="试验",
        description="保留试验",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的实验",
        replacement="实验",
        description="保留实验",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的实践",
        replacement="实践",
        description="保留实践",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的操作",
        replacement="操作",
        description="保留操作",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的执行",
        replacement="执行",
        description="保留执行",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的实施",
        replacement="实施",
        description="保留实施",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的落实",
        replacement="落实",
        description="保留落实",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的推进",
        replacement="推进",
        description="保留推进",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的推动",
        replacement="推动",
        description="保留推动",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的促进",
        replacement="促进",
        description="保留促进",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的助力",
        replacement="助力",
        description="保留助力",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的帮助",
        replacement="帮助",
        description="保留帮助",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的支持",
        replacement="支持",
        description="保留支持",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的协助",
        replacement="协助",
        description="保留协助",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的配合",
        replacement="配合",
        description="保留配合",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的协作",
        replacement="协作",
        description="保留协作",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的合作",
        replacement="合作",
        description="保留合作",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的交流",
        replacement="交流",
        description="保留交流",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的沟通",
        replacement="沟通",
        description="保留沟通",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的对话",
        replacement="对话",
        description="保留对话",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的讨论",
        replacement="讨论",
        description="保留讨论",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的探讨",
        replacement="探讨",
        description="保留探讨",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的研究",
        replacement="研究",
        description="保留研究",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的探索",
        replacement="探索",
        description="保留探索",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的发现",
        replacement="发现",
        description="保留发现",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的发明",
        replacement="发明",
        description="保留发明",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的创造",
        replacement="创造",
        description="保留创造",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的创新",
        replacement="创新",
        description="保留创新",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的突破",
        replacement="突破",
        description="保留突破",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的进展",
        replacement="进展",
        description="保留进展",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的进步",
        replacement="进步",
        description="保留进步",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的发展",
        replacement="发展",
        description="保留发展",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的成长",
        replacement="成长",
        description="保留成长",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的壮大",
        replacement="壮大",
        description="保留壮大",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的扩张",
        replacement="扩张",
        description="保留扩张",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的扩展",
        replacement="扩展",
        description="保留扩展",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的延伸",
        replacement="延伸",
        description="保留延伸",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的拓展",
        replacement="拓展",
        description="保留拓展",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的开拓",
        replacement="开拓",
        description="保留开拓",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的开发",
        replacement="开发",
        description="保留开发",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的建设",
        replacement="建设",
        description="保留建设",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的构建",
        replacement="构建",
        description="保留构建",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的打造",
        replacement="打造",
        description="保留打造",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的塑造",
        replacement="塑造",
        description="保留塑造",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的形成",
        replacement="形成",
        description="保留形成",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的产生",
        replacement="产生",
        description="保留产生",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的出现",
        replacement="出现",
        description="保留出现",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的诞生",
        replacement="诞生",
        description="保留诞生",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的问世",
        replacement="问世",
        description="保留问世",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的面世",
        replacement="面世",
        description="保留面世",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的上市",
        replacement="上市",
        description="保留上市",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的发布",
        replacement="发布",
        description="保留发布",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的推出",
        replacement="推出",
        description="保留推出",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的推出",
        replacement="推出",
        description="保留推出",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的发表",
        replacement="发表",
        description="保留发表",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的公布",
        replacement="公布",
        description="保留公布",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的公开",
        replacement="公开",
        description="保留公开",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的披露",
        replacement="披露",
        description="保留披露",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的透露",
        replacement="透露",
        description="保留透露",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的泄露",
        replacement="泄露",
        description="保留泄露",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的暴露",
        replacement="暴露",
        description="保留暴露",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的展现",
        replacement="展现",
        description="保留展现",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的展示",
        replacement="展示",
        description="保留展示",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的呈现",
        replacement="呈现",
        description="保留呈现",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的显示",
        replacement="显示",
        description="保留显示",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的表现",
        replacement="表现",
        description="保留表现",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的反映",
        replacement="反映",
        description="保留反映",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的体现",
        replacement="体现",
        description="保留体现",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的彰显",
        replacement="彰显",
        description="保留彰显",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的凸显",
        replacement="凸显",
        description="保留凸显",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的突出",
        replacement="突出",
        description="保留突出",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的显著",
        replacement="显著",
        description="保留显著",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的明显",
        replacement="明显",
        description="保留明显",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的清晰",
        replacement="清晰",
        description="保留清晰",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的明确",
        replacement="明确",
        description="保留明确",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的确定",
        replacement="确定",
        description="保留确定",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的肯定",
        replacement="肯定",
        description="保留肯定",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的认可",
        replacement="认可",
        description="保留认可",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的认同",
        replacement="认同",
        description="保留认同",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的赞同",
        replacement="赞同",
        description="保留赞同",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的支持",
        replacement="支持",
        description="保留支持",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的拥护",
        replacement="拥护",
        description="保留拥护",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的拥护",
        replacement="拥护",
        description="保留拥护",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的拥护",
        replacement="拥护",
        description="保留拥护",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的拥护",
        replacement="拥护",
        description="保留拥护",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的拥护",
        replacement="拥护",
        description="保留拥护",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的拥护",
        replacement="拥护",
        description="保留拥护",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的拥护",
        replacement="拥护",
        description="保留拥护",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的拥护",
        replacement="拥护",
        description="保留拥护",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的拥护",
        replacement="拥护",
        description="保留拥护",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的拥护",
        replacement="拥护",
        description="保留拥护",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的拥护",
        replacement="拥护",
        description="保留拥护",
        language="zh",
        priority=3,
    ),
    CompressionRule(
        pattern=r"的拥护",
        replacement="拥护",
        description="保留拥护",
        language="zh",
        priority=3,
    ),
]

EN_COMPRESSION_RULES = [
    CompressionRule(
        pattern=r"\bIn order to\b",
        replacement="To",
        description="Simplify 'In order to' to 'To'",
        language="en",
        priority=1,
    ),
    CompressionRule(
        pattern=r"\bDue to the fact that\b",
        replacement="Because",
        description="Simplify 'Due to the fact that' to 'Because'",
        language="en",
        priority=1,
    ),
    CompressionRule(
        pattern=r"\bIn spite of the fact that\b",
        replacement="Although",
        description="Simplify 'In spite of the fact that' to 'Although'",
        language="en",
        priority=1,
    ),
    CompressionRule(
        pattern=r"\bWith regard to\b",
        replacement="About",
        description="Simplify 'With regard to' to 'About'",
        language="en",
        priority=1,
    ),
    CompressionRule(
        pattern=r"\bIn relation to\b",
        replacement="About",
        description="Simplify 'In relation to' to 'About'",
        language="en",
        priority=1,
    ),
    CompressionRule(
        pattern=r"\bIn the event that\b",
        replacement="If",
        description="Simplify 'In the event that' to 'If'",
        language="en",
        priority=1,
    ),
    CompressionRule(
        pattern=r"\bFor the purpose of\b",
        replacement="To",
        description="Simplify 'For the purpose of' to 'To'",
        language="en",
        priority=1,
    ),
    CompressionRule(
        pattern=r"\bAt this point in time\b",
        replacement="Now",
        description="Simplify 'At this point in time' to 'Now'",
        language="en",
        priority=1,
    ),
    CompressionRule(
        pattern=r"\bIn the near future\b",
        replacement="Soon",
        description="Simplify 'In the near future' to 'Soon'",
        language="en",
        priority=1,
    ),
    CompressionRule(
        pattern=r"\bIn the process of\b",
        replacement="",
        description="Remove 'In the process of'",
        language="en",
        priority=2,
    ),
    CompressionRule(
        pattern=r"\bIt is important to note that\b",
        replacement="Note:",
        description="Simplify 'It is important to note that' to 'Note:'",
        language="en",
        priority=1,
    ),
    CompressionRule(
        pattern=r"\bIt should be noted that\b",
        replacement="Note:",
        description="Simplify 'It should be noted that' to 'Note:'",
        language="en",
        priority=1,
    ),
    CompressionRule(
        pattern=r"\bThere is a need to\b",
        replacement="Need to",
        description="Simplify 'There is a need to' to 'Need to'",
        language="en",
        priority=1,
    ),
    CompressionRule(
        pattern=r"\bIn addition to\b",
        replacement="Besides",
        description="Simplify 'In addition to' to 'Besides'",
        language="en",
        priority=2,
    ),
    CompressionRule(
        pattern=r"\bOn the other hand\b",
        replacement="But",
        description="Simplify 'On the other hand' to 'But'",
        language="en",
        priority=1,
    ),
    CompressionRule(
        pattern=r"\bAs a matter of fact\b",
        replacement="Actually",
        description="Simplify 'As a matter of fact' to 'Actually'",
        language="en",
        priority=1,
    ),
    CompressionRule(
        pattern=r"\bIn conclusion\b",
        replacement="So",
        description="Simplify 'In conclusion' to 'So'",
        language="en",
        priority=1,
    ),
    CompressionRule(
        pattern=r"\bTo sum up\b",
        replacement="So",
        description="Simplify 'To sum up' to 'So'",
        language="en",
        priority=1,
    ),
    CompressionRule(
        pattern=r"\bAll in all\b",
        replacement="So",
        description="Simplify 'All in all' to 'So'",
        language="en",
        priority=1,
    ),
    CompressionRule(
        pattern=r"\bFirst of all\b",
        replacement="First",
        description="Simplify 'First of all' to 'First'",
        language="en",
        priority=1,
    ),
    CompressionRule(
        pattern=r"\bLast but not least\b",
        replacement="Finally",
        description="Simplify 'Last but not least' to 'Finally'",
        language="en",
        priority=1,
    ),
    CompressionRule(
        pattern=r"\bIn other words\b",
        replacement="Meaning",
        description="Simplify 'In other words' to 'Meaning'",
        language="en",
        priority=2,
    ),
    CompressionRule(
        pattern=r"\bFor example\b",
        replacement="Like",
        description="Simplify 'For example' to 'Like'",
        language="en",
        priority=2,
    ),
    CompressionRule(
        pattern=r"\bFor instance\b",
        replacement="Like",
        description="Simplify 'For instance' to 'Like'",
        language="en",
        priority=2,
    ),
    CompressionRule(
        pattern=r"\bSuch as\b",
        replacement="Like",
        description="Simplify 'Such as' to 'Like'",
        language="en",
        priority=2,
    ),
    CompressionRule(
        pattern=r"\bIn terms of\b",
        replacement="For",
        description="Simplify 'In terms of' to 'For'",
        language="en",
        priority=2,
    ),
    CompressionRule(
        pattern=r"\bWith respect to\b",
        replacement="About",
        description="Simplify 'With respect to' to 'About'",
        language="en",
        priority=2,
    ),
    CompressionRule(
        pattern=r"\bIn reference to\b",
        replacement="About",
        description="Simplify 'In reference to' to 'About'",
        language="en",
        priority=2,
    ),
    CompressionRule(
        pattern=r"\bIn connection with\b",
        replacement="About",
        description="Simplify 'In connection with' to 'About'",
        language="en",
        priority=2,
    ),
    CompressionRule(
        pattern=r"\bIn accordance with\b",
        replacement="Per",
        description="Simplify 'In accordance with' to 'Per'",
        language="en",
        priority=2,
    ),
    CompressionRule(
        pattern=r"\bIn compliance with\b",
        replacement="Per",
        description="Simplify 'In compliance with' to 'Per'",
        language="en",
        priority=2,
    ),
    CompressionRule(
        pattern=r"\bIn conformity with\b",
        replacement="Per",
        description="Simplify 'In conformity with' to 'Per'",
        language="en",
        priority=2,
    ),
    CompressionRule(
        pattern=r"\bIn line with\b",
        replacement="Per",
        description="Simplify 'In line with' to 'Per'",
        language="en",
        priority=2,
    ),
    CompressionRule(
        pattern=r"\bIn keeping with\b",
        replacement="Per",
        description="Simplify 'In keeping with' to 'Per'",
        language="en",
        priority=2,
    ),
    CompressionRule(
        pattern=r"\bIn keeping with\b",
        replacement="Per",
        description="Simplify 'In keeping with' to 'Per'",
        language="en",
        priority=2,
    ),
]


class SemanticCompressor:
    """Compresses text by using shorter expressions while preserving meaning."""

    def __init__(self):
        self.zh_rules = sorted(ZH_COMPRESSION_RULES, key=lambda r: r.priority)
        self.en_rules = sorted(EN_COMPRESSION_RULES, key=lambda r: r.priority)

    def detect_language(self, text: str) -> str:
        """Detect if text is primarily Chinese or English."""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len(text.replace(" ", "").replace("\n", ""))
        if total_chars == 0:
            return "zh"
        ratio = chinese_chars / total_chars
        return "zh" if ratio > 0.3 else "en"

    def compress(self, text: str, language: Optional[str] = None) -> CompressionResult:
        """Compress text using semantic compression rules.

        Args:
            text: Original text to compress
            language: Force language ('zh' or 'en'), auto-detect if None

        Returns:
            CompressionResult with compressed text and metadata
        """
        if not text:
            return CompressionResult(
                original_text="",
                compressed_text="",
                original_length=0,
                compressed_length=0,
                compression_ratio=1.0,
                rules_applied=[],
                token_reduction=0,
            )

        lang = language or self.detect_language(text)
        rules = self.zh_rules if lang == "zh" else self.en_rules

        compressed = text
        applied_rules = []

        for rule in rules:
            new_text = re.sub(rule.pattern, rule.replacement, compressed)
            if new_text != compressed:
                compressed = new_text
                applied_rules.append(rule.description)

        original_length = len(text)
        compressed_length = len(compressed)
        compression_ratio = compressed_length / original_length if original_length > 0 else 1.0
        token_reduction = original_length - compressed_length

        compressed = self._clean_whitespace(compressed)

        return CompressionResult(
            original_text=text,
            compressed_text=compressed,
            original_length=original_length,
            compressed_length=compressed_length,
            compression_ratio=round(compression_ratio, 3),
            rules_applied=applied_rules,
            token_reduction=token_reduction,
            meaning_preserved=True,
        )

    def compress_messages(self, messages: list[dict], language: Optional[str] = None) -> list[dict]:
        """Compress a list of messages.

        Args:
            messages: List of message dicts with 'role' and 'content'
            language: Force language, auto-detect per message if None

        Returns:
            List of compressed messages
        """
        compressed_messages = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > 50:
                result = self.compress(content, language)
                if result.compression_ratio < 0.95:
                    compressed_messages.append({
                        **msg,
                        "content": result.compressed_text,
                        "metadata": {
                            **msg.get("metadata", {}),
                            "compressed": True,
                            "original_length": result.original_length,
                            "compressed_length": result.compressed_length,
                            "compression_ratio": result.compression_ratio,
                        },
                    })
                else:
                    compressed_messages.append(msg)
            else:
                compressed_messages.append(msg)
        return compressed_messages

    def _clean_whitespace(self, text: str) -> str:
        """Clean up excessive whitespace after compression."""
        text = re.sub(r' {2,}', ' ', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'，{2,}', '，', text)
        text = re.sub(r', {2,}', ', ', text)
        text = text.strip()
        return text

    def estimate_token_savings(self, text: str) -> dict:
        """Estimate token savings from compression."""
        result = self.compress(text)
        original_tokens = self._estimate_tokens(text)
        compressed_tokens = self._estimate_tokens(result.compressed_text)
        return {
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "token_savings": original_tokens - compressed_tokens,
            "savings_percentage": round(
                (original_tokens - compressed_tokens) / original_tokens * 100, 2
            ) if original_tokens > 0 else 0,
        }

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return int(other_chars / 4 + chinese_chars / 1.5)
