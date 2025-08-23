import pytest
from dotenv import load_dotenv
from deepeval import assert_test
from deepeval.metrics import HallucinationMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.dataset import EvaluationDataset

load_dotenv()

first_test_case = LLMTestCase(
    input="Pythonは1991年にGuido van Rossumによって開発された高水準プログラミング言語です。", 
    actual_output="Pythonは1991年にGuido van Rossumによって開発された高水準プログラミング言語で、読みやすさと簡潔性で人気があります。")
second_test_case = LLMTestCase(
    input="日本の首都は東京です。", 
    actual_output="日本の首都は東京で、経済、文化、政治の中心地です。")

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