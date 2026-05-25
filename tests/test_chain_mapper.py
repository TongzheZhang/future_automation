"""
测试产业链映射模块
"""

import pytest
from research.chain_mapper import ChainMapper, COMMODITY_CHAINS


def test_get_chain_by_commodity():
    """测试根据品种找产业链"""
    mapper = ChainMapper()
    
    # 螺纹钢属于黑色系
    assert mapper.get_chain_by_commodity("RB") == "black"
    # 铜属于铜产业链
    assert mapper.get_chain_by_commodity("CU") == "copper"
    # 不存在的品种
    assert mapper.get_chain_by_commodity("UNKNOWN") is None


def test_get_related_commodities():
    """测试获取相关品种"""
    mapper = ChainMapper()
    
    related = mapper.get_related_commodities("RB")
    assert "I" in related  # 铁矿石
    assert "J" in related  # 焦炭


def test_get_chain_description():
    """测试产业链描述"""
    mapper = ChainMapper()
    desc = mapper.get_chain_description("black")
    
    assert "黑色系" in desc
    assert "铁矿石" in desc
    assert "钢材" in desc
