"""제공 data.json을 수정하지 않고 케이스별 검증과 판정을 수행한다."""
import json
from pathlib import Path
import re

from core import Matrix, mac, normalize_label, compare_scores


def load_document(path):
    with Path(path).open(encoding='utf-8') as file:
        document = json.load(file)
    if not isinstance(document, dict):
        raise ValueError('최상위 JSON은 객체여야 합니다.')
    if not isinstance(document.get('filters'), dict):
        raise ValueError('filters 객체가 필요합니다.')
    if not isinstance(document.get('patterns'), dict) or not document['patterns']:
        raise ValueError('비어 있지 않은 patterns 객체가 필요합니다.')
    return document


def validate_filters(raw, size):
    if not isinstance(raw, dict):
        raise ValueError('size_{} 필터 객체가 없습니다.'.format(size))
    normalized = {}
    for label, rows in raw.items():
        name = normalize_label(label)
        if name in normalized:
            raise ValueError('정규화 후 중복 필터 라벨: ' + name)
        normalized[name] = Matrix(rows, size)
    if set(normalized) != {'Cross', 'X'}:
        raise ValueError('Cross와 X 필터가 각각 하나씩 필요합니다.')
    return normalized


def analyze(document):
    """한 케이스의 오류는 다른 케이스의 분석을 중단하지 않는다."""
    results = []
    cache = {}
    for identifier, item in document['patterns'].items():
        result = {'id': identifier, 'passed': False, 'prediction': None,
                  'expected': None, 'scores': None, 'reason': '', 'category': ''}
        try:
            match = re.fullmatch(r'size_([1-9][0-9]*)_([0-9]+)', identifier)
            if not match:
                raise ValueError('키 형식 오류: size_{N}_{idx} 필요')
            size = int(match.group(1))
            if size not in cache:
                cache[size] = validate_filters(document['filters'].get('size_{}'.format(size)), size)
            if not isinstance(item, dict) or 'input' not in item or 'expected' not in item:
                raise ValueError('각 패턴에 input과 expected가 필요합니다.')
            expected = normalize_label(item['expected'])
            result['expected'] = expected
            pattern = Matrix(item['input'], size)
            filters = cache[size]
            scores = {label: mac(pattern, filters[label]) for label in ('Cross', 'X')}
            prediction = compare_scores(scores['Cross'], scores['X'])
            result.update(scores=scores, prediction=prediction, passed=prediction == expected)
            if prediction == 'UNDECIDED':
                result.update(reason='동점: |Cross-X| < 1e-9, expected와 일치하지 않음', category='수치 비교')
            elif not result['passed']:
                result.update(reason='점수 우세 라벨 {}와 expected {} 불일치'.format(prediction, expected), category='데이터/분류 결과')
        except (ValueError, TypeError, KeyError, OverflowError, ArithmeticError) as error:
            result.update(reason=str(error), category='데이터/스키마 또는 수치 범위')
        results.append(result)
    return results


def summarize(results):
    passed = sum(result['passed'] for result in results)
    return len(results), passed, len(results) - passed
