import pytest
from deepeval import assert_test
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

first_test_case = LLMTestCase(
    input="Pythonは1991年にGuido van Rossumによって開発された高水準プログラミング言語です。", 
    actual_output="Pythonは1991年にGuido van Rossumによって開発された高水準プログラミング言語で、読みやすさと簡潔性で人気があります。")
second_test_case = LLMTestCase(
    input="パーキンソン病の治療には複数の薬剤が用いられます。レボドパは運動症状の改善に効果的です。", 
    actual_output="パーキンソン病の治療には複数の薬剤が用いられます。レボドパは運動症状の改善に効果的で、副作用として消化器症状や眠気があります。")

dataset = EvaluationDataset()  # 空のインスタンス生成

dataset.test_cases.append(first_test_case)
dataset.test_cases.append(second_test_case)

@pytest.mark.parametrize(
    "test_case",
    dataset.test_cases,
)
def test_example(test_case: LLMTestCase):
    metric = AnswerRelevancyMetric(threshold=0.5)
    assert_test(test_case, [metric])