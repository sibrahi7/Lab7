from typing import Dict, List
import pytest
import copy
from unittest import mock

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import (
    InvalidParamError,
    RecognizerResult,
    OperatorConfig,
    PIIEntity,
    OperatorResult,
    EngineResult,
)
from presidio_anonymizer.operators import OperatorType, AHDS_AVAILABLE


def test_given_request_anonymizers_return_list():
    engine = AnonymizerEngine()
    expected_list = {"hash", "mask", "redact", "replace", "custom", "keep", "encrypt"}
    if AHDS_AVAILABLE:
        expected_list.add("surrogate_ahds")
    anon_list = set(engine.get_anonymizers())

    assert anon_list == expected_list


def test_given_empty_analyzers_list_then_we_get_same_text_back():
    engine = AnonymizerEngine()
    text = "one two three"
    assert engine.anonymize(text, [], {}).text == text


def test_given_empty_anonymziers_list_then_we_fall_to_default():
    engine = AnonymizerEngine()
    text = "please REPLACE ME."
    analyzer_result = RecognizerResult("SSN", 7, 17, 0.8)
    result = engine.anonymize(text, [analyzer_result], {}).text
    assert result == "please <SSN>."


def test_given_custom_anonymizer_then_we_manage_to_anonymize_successfully():
    engine = AnonymizerEngine()
    text = (
        "Fake card number 4151 3217 6243 3448.com that "
        "overlaps with nonexisting URL."
    )
    analyzer_result = RecognizerResult("CREDIT_CARD", 17, 36, 0.8)
    analyzer_result2 = RecognizerResult("URL", 32, 40, 0.8)
    anonymizer_config = OperatorConfig("custom", {"lambda": lambda x: f"<ENTITY: {x}>"})
    result = engine.anonymize(
        text, [analyzer_result, analyzer_result2], {"DEFAULT": anonymizer_config}
    ).text
    resp = (
        "Fake card number <ENTITY: 4151 3217 6243 3448>"
        "<ENTITY: 3448.com> that overlaps with nonexisting URL."
    )
    assert result == resp


def test_given_none_as_anonymziers_list_then_we_fall_to_default():
    engine = AnonymizerEngine()
    text = "please REPLACE ME."
    analyzer_result = RecognizerResult("SSN", 7, 17, 0.8)
    result = engine.anonymize(text, [analyzer_result]).text
    assert result == "please <SSN>."


def test_given_default_anonymizer_then_we_use_it():
    engine = AnonymizerEngine()
    text = "please REPLACE ME."
    analyzer_result = RecognizerResult("SSN", 7, 17, 0.8)
    anonymizer_config = OperatorConfig("replace", {"new_value": "and thank you"})
    result = engine.anonymize(
        text, [analyzer_result], {"DEFAULT": anonymizer_config}
    ).text
    assert result == "please and thank you."


def test_given_specific_anonymizer_then_we_use_it():
    engine = AnonymizerEngine()
    text = "please REPLACE ME."
    analyzer_result = RecognizerResult("SSN", 7, 17, 0.8)
    anonymizer_config = OperatorConfig("replace", {"new_value": "and thank you"})
    ssn_anonymizer_config = OperatorConfig("redact", {})
    result = engine.anonymize(
        text,
        [analyzer_result],
        {"DEFAULT": anonymizer_config, "SSN": ssn_anonymizer_config},
    ).text
    assert result == "please ."


@pytest.mark.parametrize(
    "original_text,start,end",
    [
        ("hello world", 5, 12),
        ("hello world", 12, 16),
    ],
)
def test_given_analyzer_result_with_an_incorrect_text_positions_then_we_fail(
    original_text, start, end
):
    engine = AnonymizerEngine()
    analyzer_result = RecognizerResult("type", start, end, 0.5)
    err_msg = (
        f"Invalid analyzer result, start: {start} and end: "
        f"{end}, while text length is only 11."
    )
    with pytest.raises(InvalidParamError, match=err_msg):
        engine.anonymize(original_text, [analyzer_result], {})


@pytest.mark.parametrize(
    "anonymizers, result_text",
    [
        ({"number": OperatorConfig("fake")}, "Invalid operator class 'fake'."),
    ],
)
def test_given_invalid_json_for_anonymizers_then_we_fail(anonymizers, result_text):
    with pytest.raises(InvalidParamError, match=result_text):
        AnonymizerEngine().anonymize(
            "this is my text", [RecognizerResult("number", 0, 4, 0)], anonymizers
        )


def test_given_analyzer_result_input_then_it_is_not_mutated():
    engine = AnonymizerEngine()
    text = "Jane Doe is a person"
    original_analyzer_results = [
        RecognizerResult(start=0, end=4, entity_type="PERSON", score=1.0),
        RecognizerResult(start=5, end=8, entity_type="PERSON", score=1.0),
    ]
    copy_analyzer_results = copy.deepcopy(original_analyzer_results)
    engine.anonymize(text, original_analyzer_results)

    assert original_analyzer_results == copy_analyzer_results


def test_given_unsorted_input_then_merged_correctly():
    engine = AnonymizerEngine()
    text = "Jane Doe is a person"
    original_analyzer_results = [
        RecognizerResult(start=5, end=8, entity_type="PERSON", score=1.0),
        RecognizerResult(start=0, end=4, entity_type="PERSON", score=1.0),
    ]
    anonymizer_result = engine.anonymize(text, original_analyzer_results)
    assert anonymizer_result.text == "<PERSON> is a person"


@mock.patch("presidio_anonymizer.anonymizer_engine.logger")
def test_given_conflict_input_then_merged_correctly(mock_logger):
    engine = AnonymizerEngine()
    text = "I'm George Washington Square Park."
    original_analyzer_results = [
        RecognizerResult(start=4, end=21, entity_type="PERSON", score=1.0),
        RecognizerResult(start=4, end=33, entity_type="LOCATION", score=1.0),
    ]

    anonymizer_result = engine.anonymize(
        text,
        original_analyzer_results
    )

    assert anonymizer_result.text == "I'm <LOCATION>."
    mock_logger.debug.assert_called()


def _operate(
    text: str,
    pii_entities: List[PIIEntity],
    operators_metadata: Dict[str, OperatorConfig],
    operator_type: OperatorType,
) -> EngineResult:
    return EngineResult(
        "Number: I am your new text!", [OperatorResult(0, 35, "type", "text", "hash")]
    )
