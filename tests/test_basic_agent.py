from src.agents.basic_agent import BasicAgent

def test_act_returns_zero():
    agent = BasicAgent()
    assert agent.act([]) == 0
